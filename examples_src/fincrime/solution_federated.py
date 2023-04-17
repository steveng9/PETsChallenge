from pathlib import Path
from typing import Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import FitIns, FitRes, Parameters
from flwr.server import ClientManager
from flwr.server.client_proxy import ClientProxy
from loguru import logger
import numpy as np
import pandas as pd
import os
import joblib
import pickle
from sklearn import metrics
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from federated_fraud_detection import key_gen, build_okvs, initiate_queries, randomize, combine, decrypt, finish

from .model import SwiftModel


def empty_parameters() -> Parameters:
    """Utility function that generates empty Flower Parameters dataclass instance."""
    return fl.common.ndarrays_to_parameters([])


def encrypt_banks(banks, key):
    nonce = os.urandom(12)
    data = ','.join(banks)
    data_byte = bytes(data, 'utf-8')
    cipher = AESGCM(key)
    return cipher.encrypt(nonce, data_byte, b""), nonce


def decrypt_banks(ciphertext, nonce, key):
    cipher = AESGCM(key)
    data = cipher.decrypt(nonce, ciphertext, b"").decode()
    return data.split(",")


def test_setup(server_dir: Path, client_dirs_dict: Dict[str, Path]):
    for client in client_dirs_dict.keys():
        if client != 'swift':
            key = AESGCM.generate_key(bit_length=128)
            joblib.dump(key, client_dirs_dict[client] / 'symmetric_key.joblib')
            joblib.dump(key, client_dirs_dict["swift"] / (client + '_symmetric_key.joblib'))


# ___________________________________________________________________________________________________________


class TrainingSwiftClient(fl.client.NumPyClient):
    def __init__(
        self, cid: str, swift_df: pd.DataFrame, model: SwiftModel, client_dir: Path
    ):
        super().__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.model = model
        self.client_dir = client_dir

    def fit(self, parameters: List[np.ndarray], config: dict):
        logger.info("SWIFT {}, round {}", self.cid, config["round"])
        self.model.fit(self.swift_df)
        logger.info("SWIFT saving model to disk...")
        self.model.save(self.client_dir)
        return [], 1, {}


class TrainingBankClient(fl.client.NumPyClient):
    def __init__(self, cid: str):
        super().__init__()
        self.cid = cid

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        logger.info("Bank {}, round {}", self.cid, config["round"])
        return [], 1, {}


def train_client_factory(cid, data_path: Path, client_dir: Path):
    if cid == "swift":
        logger.info("Initializing SWIFT client for '{}'", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        model = SwiftModel()
        return TrainingSwiftClient(
            cid, swift_df=swift_df, model=model, client_dir=client_dir
        )
    else:
        logger.info("Initializing bank client for {}", cid)
        return TrainingBankClient(cid)


class TrainStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        self.client_banks_dict = {}
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        return empty_parameters()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}
        if server_round == 1:
            fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            fit_config: List[Tuple[ClientProxy, FitIns]] = [(client_dict["swift"], fit_ins)]
            return fit_config

    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        if (n_failures := len(failures)) > 0:
            raise Exception(f"Had {n_failures} failures in round {server_round}")

        for client, result in results:
            result_ndarrays = fl.common.parameters_to_ndarrays(result.parameters)
            logger.info("aggregate client {}, round {}", client.cid, server_round)
        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        return None

    def evaluate(self, server_round, parameters):
        return None


def train_strategy_factory(server_dir: Path):
    training_strategy = TrainStrategy(server_dir=server_dir)
    num_rounds = 1
    return training_strategy, num_rounds


#______________________________________________________________________________


def test_client_factory(
    cid: str,
    data_path: Path,
    client_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    if cid == "swift":
        logger.info("Initializing SWIFT Test client for {}", cid)
        swift_df = pd.read_csv(data_path, index_col="MessageId")
        return TestSwiftClient(
            cid,
            swift_df=swift_df,
            client_dir=client_dir,
            preds_format_path=preds_format_path,
            preds_dest_path=preds_dest_path,
        )
    else:
        logger.info("Initializing bank test client for {}", cid)
        bank_df = pd.read_csv(data_path, dtype=pd.StringDtype())
        return TestBankClient(cid, bank_df=bank_df, client_dir=client_dir)



class TestSwiftClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        swift_df: pd.DataFrame,
        client_dir: Path,
        preds_format_path: Path,
        preds_dest_path: Path,
    ):
        super().__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.client_dir = client_dir
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path
        self.ensemble_df = pd.DataFrame()
        self.messages = []

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        round = config["round"]
        del config["round"]

        if round == 1:
            raise Exception("SWIFT does not participate in round 1")
        elif round == 2:
            # save mapping of clients to banks
            client_to_banks = {}
            # decrypt bank mappings sent from each client
            for (client, banks_encrypted, nonce) in zip(*parameters):
                key = joblib.load(self.client_dir / (client + '_symmetric_key.joblib'))
                client_to_banks[client] = decrypt_banks(banks_encrypted, nonce, key)
            bank_to_client = {}
            for client, banks in client_to_banks.items():
                for bank in banks:
                    bank_to_client[bank] = client
            joblib.dump(client_to_banks, self.client_dir / 'client_to_banks.joblib')
            joblib.dump(bank_to_client, self.client_dir / 'bank_to_client.joblib')

            pairs_banks = self.swift_df.loc[:,['Sender', 'Receiver']].drop_duplicates()
            pairs_banks['sending_client'] = pairs_banks['Sender'].apply(lambda x: bank_to_client.get(x))
            pairs_banks['receiving_client'] = pairs_banks['Receiver'].apply(lambda x: bank_to_client.get(x))
            # don't include pairs where bank does not belong to any client
            all_sending_receiving_pairs_clients = pairs_banks.loc[
                pairs_banks['sending_client'].notnull() &
                pairs_banks['receiving_client'].notnull(),
                ['sending_client', 'receiving_client']].drop_duplicates().values
            joblib.dump(all_sending_receiving_pairs_clients, self.client_dir / 'all_sending_receiving_pairs.joblib')

            # create mapping of clients to their okvs and public keys
            client_keys = {}
            client_okvs = {}
            joblib.dump(client_keys, self.client_dir / 'client_keys.joblib')
            joblib.dump(client_okvs, self.client_dir / 'client_okvs.joblib')
            return [], 1, {}

        elif config["okvs protocol"] == "step1":
            client_keys = joblib.load(self.client_dir / "client_keys.joblib")
            client_okvs = joblib.load(self.client_dir / "client_okvs.joblib")

            # For one round per node, receive and save to file the okvs and keys they've sent. First load
            # the file, which was created in prior round, then append to it for next client.
            client = parameters[0][0]
            client_keys[client] = parameters[1] # pk
            client_okvs[client] = parameters[2] # okvs

            joblib.dump(client_keys, self.client_dir / 'client_keys.joblib')
            joblib.dump(client_okvs, self.client_dir / 'client_okvs.joblib')

            return [], 1, {}

        elif config["okvs protocol"] == "step2":
            all_sending_receiving_pairs = joblib.load(self.client_dir / 'all_sending_receiving_pairs.joblib')
            client_to_banks = joblib.load(self.client_dir / 'client_to_banks.joblib')
            client_okvss = joblib.load(self.client_dir / 'client_okvs.joblib')

            # create ciphertexts for every sending/receiving client pair, and send it to them.
            client_pairing_swift_keys = {}
            senders = []
            receivers = []
            ciphertexts = []
            for sndr_rcvr in all_sending_receiving_pairs:
                sender = sndr_rcvr[0]
                sender_banks = client_to_banks[sender]
                receiver = sndr_rcvr[1]
                receiver_banks = client_to_banks[receiver]
                logger.info("creating ciphertexts for {} and {}", sender, receiver)
                pk, sk = key_gen()
                client_pairing_swift_keys[(sender, receiver)] = (pk, sk)
                sender_okvs = client_okvss[sender]
                receiver_okvs = client_okvss[receiver]
                data = self.swift_df[(self.swift_df['Sender'].isin(sender_banks)) & (self.swift_df['Receiver'].isin(receiver_banks))]
                sender_data = data[['OrderingAccount', 'OrderingName', 'OrderingStreet', 'OrderingCountryCityZip']].values.astype("U")
                receiver_data = data[['BeneficiaryAccount', 'BeneficiaryName', 'BeneficiaryStreet', 'BeneficiaryCountryCityZip']].values.astype("U")

                ciphertext = initiate_queries(sender_data, receiver_data, sender_okvs.tolist(), receiver_okvs.tolist(), pk)

                senders.append(sender)
                receivers.append(receiver)
                ciphertexts.append(np.array(ciphertext, dtype=np.uint8))

            params = [np.array([pickle.dumps([senders, receivers, ciphertexts], protocol=5)])]

            joblib.dump(client_pairing_swift_keys, self.client_dir / 'client_pairing_swift_keys.joblib')
            return params, 1, {}

        elif config["okvs protocol"] == "step4":
            senders = []
            receivers = []
            As = []
            Bs = []
            sndr_rcvr_c_d = {}
            logger.info("swift, step4")
            params = pickle.loads(parameters[0][0])
            for (sndr, rcvr, sndr_randomized, rcvr_randomized) in zip(*params):
                a, b, c, d = combine(sndr_randomized.tolist(), rcvr_randomized.tolist())
                sndr_rcvr_c_d[(sndr, rcvr)] = (c, d)

                senders.append(sndr)
                receivers.append(rcvr)
                As.append(np.array(a, dtype=np.uint8))
                Bs.append(np.array(b, dtype=np.uint8))

            joblib.dump(sndr_rcvr_c_d, self.client_dir / 'sndr_rcvr_c_d.joblib')

            return [np.array([pickle.dumps([senders, receivers, As, Bs], protocol=5)])], 1, {}

        elif config["okvs protocol"] == "step6":
            logger.info("swift step6, finishing OKVS protocol.")
            # Finish OKVS protocol. Now able to determine validity of transactions
            params = pickle.loads(parameters[0][0])
            sndr_rcvr_c_d = joblib.load(self.client_dir / "sndr_rcvr_c_d.joblib")
            client_pairing_swift_keys = joblib.load(self.client_dir / 'client_pairing_swift_keys.joblib')
            client_to_banks = joblib.load(self.client_dir / 'client_to_banks.joblib')
            self.swift_df['Bank_Validity'] = False # default to False if no output matched from bank client

            sndr_rcvr_outputs = {}
            for (sndr, rcvr, sndr_decrypted, rcvr_decrypted) in zip(*params):
                pair = (sndr, rcvr)
                c, d = sndr_rcvr_c_d[pair]
                secret_key_for_pair = client_pairing_swift_keys[pair][1]
                outputs = finish(sndr_decrypted.tolist(), rcvr_decrypted.tolist(), c, d, secret_key_for_pair)
                sndr_rcvr_outputs[pair] = outputs

                sender_banks = client_to_banks[sndr]
                receiver_banks = client_to_banks[rcvr]
                self.swift_df.loc[
                    (self.swift_df['Sender'].isin(sender_banks)) & (self.swift_df['Receiver'].isin(receiver_banks)), 'Bank_Validity'
                ] = outputs

            # make sure all rows received output from protocol
            joblib.dump(sndr_rcvr_outputs, self.client_dir / "outputs_from_okvs_protocol.joblib")

            model = SwiftModel.load(self.client_dir)
            ensemble_df = pd.DataFrame()
            ensemble_df['SWIFT'] = model.predict(self.swift_df)
            ensemble_df['Bank'] = self.swift_df["Bank_Validity"].apply(lambda x: 0 if x else 1).values
            ensemble_df['SWIFT+Bank'] = ensemble_df[['SWIFT', 'Bank']].max(axis=1)


            print("AUPRC LGB only swift:", metrics.average_precision_score(y_true=self.swift_df["Label"].values,
                                                                y_score=ensemble_df['SWIFT'].values))
            print("AUPRC LGB w/ banking membership:", metrics.average_precision_score(y_true=self.swift_df["Label"].values,
                                                                y_score=ensemble_df['SWIFT+Bank'].values))
            print("AUPRC LGB only bank membership:", metrics.average_precision_score(y_true=self.swift_df["Label"].values,
                                                                y_score=ensemble_df['Bank'].values))

            preds_format_df = pd.read_csv(self.preds_format_path, index_col="MessageId")
            preds_format_df["Score"] = preds_format_df.assign(Score=ensemble_df['SWIFT+Bank'].values)
            preds_format_df.to_csv(self.preds_dest_path)

            return [], 1, {}




class TestBankClient(fl.client.NumPyClient):
    def __init__(self, cid, bank_df, client_dir):
        super().__init__()
        self.cid = cid
        self.bank_df = bank_df
        # self.model = model
        self.client_dir = client_dir

    def fit(
        self, parameters: List[np.ndarray], config: dict
    ) -> Tuple[List[np.ndarray], int, dict]:
        round = config["round"]
        logger.info("Bank {}, round {}", self.cid, round)
        if round == 1:
            key = joblib.load(self.client_dir / 'symmetric_key.joblib')
            message, nonce = encrypt_banks(list(self.bank_df["Bank"].unique()), key)
            return [], 1, {"message": message, "nonce": nonce}

        elif round == 2:
            logger.info("building okvs for {}", self.cid)
            pk, sk = key_gen()
            self.bank_df['Flags'] = self.bank_df['Flags'].apply(lambda x: int(x))
            okvs = build_okvs(pk, self.bank_df.loc[self.bank_df['Flags'] == 0][["Account", "Name", 'Street', 'CountryCityZip']].values.astype("U"))

            joblib.dump(sk, self.client_dir / (self.cid + "sk.joblib"))
            return [np.array(pk, dtype=np.uint8), np.array(okvs, dtype=np.uint8)], 1, {}

        elif config["okvs protocol"] == "step3":
            logger.info("In bank {}, step3", self.cid)
            parameters = pickle.loads(parameters[0][0])
            sender_ciphers = parameters[0]
            receiver_ciphers = parameters[1]
            sender_ciphers_randomized = [np.array(randomize(ciphertext.tolist()), dtype=np.uint8) for ciphertext in sender_ciphers]
            receiver_ciphers_randomized = [np.array(randomize(ciphertext.tolist()), dtype=np.uint8) for ciphertext in receiver_ciphers]
            return [np.array([pickle.dumps([sender_ciphers_randomized, receiver_ciphers_randomized], protocol=5)])], 1, {}

        elif config["okvs protocol"] == "step5":
            logger.info("In bank {}, step5", self.cid)
            secret_key = joblib.load(self.client_dir / (self.cid + "sk.joblib"))
            parameters = pickle.loads(parameters[0][0])
            As = parameters[0]
            Bs = parameters[1]
            sender_decrypteds = [np.array(decrypt(a.tolist(), secret_key), dtype=np.uint8) for a in As]
            receiver_decrypteds = [np.array(decrypt(b.tolist(), secret_key), dtype=np.uint8) for b in Bs]
            return [np.array([pickle.dumps([sender_decrypteds, receiver_decrypteds], protocol=5)])], 1, {}






def test_strategy_factory(server_dir: Path):
    test_strategy = TestStrategy(server_dir=server_dir)
    num_rounds = 17  # At most 17 rounds, if at most 10 banking clients; add 7 rounds to this
    return test_strategy, num_rounds




class TestStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        self.server_dir = server_dir
        self.swift_transactions_for_banks = None
        self.client_banks_dict = {}
        self.client_to_pk = {}
        self.client_to_okvs = {}
        self.num_bank_clients = 0
        self.ciphertexts = []
        self.sndr_rcvr = []
        self.randomized_ciphers = {}
        super().__init__()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return empty_parameters()

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        client_dict: Dict[str, ClientProxy] = client_manager.all()
        config_dict = {"round": server_round}

        if server_round == 1:
            self.num_bank_clients = len(client_dict) - 1
            bank_fit_ins = FitIns(parameters=empty_parameters(), config=config_dict)
            return [(v, bank_fit_ins) for k, v in client_dict.items() if k != "swift"]

        elif server_round == 2:
            clients = []
            banks_encrypteds = []
            nonces = []
            for client_cid, (banks_encrypted, nonce) in self.client_banks_dict.items():
                clients.append(client_cid)
                banks_encrypteds.append(banks_encrypted)
                nonces.append(nonce)
            params = [np.array(clients), np.array(banks_encrypteds), np.array(nonces)]
            fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(params), config=config_dict)
            fit_config = [(client_dict["swift"], fit_ins)]
            for k, v in client_dict.items():
                if k != "swift":
                    # wake bank by sending empty message, so they can build their OKVS
                    fit_config.append((v, FitIns(parameters=empty_parameters(), config=config_dict)))
            return fit_config

        elif server_round >= 3 and server_round - 3 < self.num_bank_clients: # subtract one to only count BANK clients
            config_dict["okvs protocol"] = "step1"
            clients = sorted(client_dict.keys())
            client_chosen = clients[server_round-3]
            logger.info("round... {}, client okvs materials being sent to swift: {}", server_round, client_chosen)
            pk = self.client_to_pk[client_chosen]
            okvs = self.client_to_okvs[client_chosen]
            parameters_array = [
                np.array([client_chosen]),
                pk, # pk
                okvs # okvs
            ]
            fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(parameters_array), config=config_dict)
            return [(client_dict["swift"], fit_ins)]

        elif server_round - 3 == self.num_bank_clients: # here swift creates all the ciphertexts
            config_dict["okvs protocol"] = "step2"
            return [(client_dict["swift"], FitIns(parameters=empty_parameters(), config=config_dict))]

        elif server_round - 4 == self.num_bank_clients: # send ciphertexts to clients
                                   # [sender ciphers, receiver ciphers]
            client_params = {client: [[], []] for client in client_dict.keys() if client != "swift"}
            fit_config = []
            config_dict["okvs protocol"] = "step3"

            for (sndr, rcvr, ciphertext) in zip(*self.ciphertexts):
                self.sndr_rcvr.append((sndr, rcvr)) # build up same ordering of sender/receiver client pairs
                client_params[sndr][0].append(ciphertext)
                client_params[rcvr][1].append(ciphertext)

            for client, params in client_params.items():
                parameters = [np.array([pickle.dumps(params, protocol=5)])]
                fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(parameters), config=config_dict)
                fit_config.append((client_dict[client], fit_ins))
            return fit_config

        elif server_round - 5 == self.num_bank_clients: # send randomized ciphertexts to swift
            config_dict["okvs protocol"] = "step4"
            # pop randomized ciphers in same order as they were sent, to assure ciphers are
            # associated with correct sender/receiver pair
            sndrs = []
            rcvrs = []
            sndr_ranomizeds = []
            rcvr_ranomizeds = []
            for sndr, rcvr in self.sndr_rcvr:
                sndrs.append(sndr)
                rcvrs.append(rcvr)
                sndr_ranomizeds.append(self.randomized_ciphers[sndr][0].pop(0))
                rcvr_ranomizeds.append(self.randomized_ciphers[rcvr][1].pop(0))

            params_array = [np.array([pickle.dumps([sndrs, rcvrs, sndr_ranomizeds, rcvr_ranomizeds], protocol=5)])]
            fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(params_array), config=config_dict)
            return [(client_dict["swift"], fit_ins)]

        elif server_round - 6 == self.num_bank_clients: # send combined randomized parts to clients
                                   # [sender a's, receiver b's]
            client_params = {client: [[], []] for client in client_dict.keys() if client != "swift"}
            fit_config = []
            config_dict["okvs protocol"] = "step5"

            self.sndr_rcvr = []
            for (sndr, rcvr, a, b) in zip(*self.ciphertexts):
                self.sndr_rcvr.append((sndr, rcvr)) # build up same ordering of sender/receiver client pairs
                client_params[sndr][0].append(a)
                client_params[rcvr][1].append(b)

            for client, params in client_params.items():
                parameters = [np.array([pickle.dumps(params, protocol=5)])]
                fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(parameters), config=config_dict)
                fit_config.append((client_dict[client], fit_ins))
            return fit_config

        elif server_round - 7 == self.num_bank_clients: # send penultimate_decryptions to swift
            config_dict["okvs protocol"] = "step6"
            # pop randomized ciphers in same order as they were sent, to assure ciphers are
            # associated with correct sender/receiver pair
            sndrs = []
            rcvrs = []
            sndr_decrypteds = []
            rcvr_decrypteds = []
            for sndr, rcvr in self.sndr_rcvr:
                sndrs.append(sndr)
                rcvrs.append(rcvr)
                sndr_decrypteds.append(self.randomized_ciphers[sndr][0].pop(0))
                rcvr_decrypteds.append(self.randomized_ciphers[rcvr][1].pop(0))
            params_array = [np.array([pickle.dumps([sndrs, rcvrs, sndr_decrypteds, rcvr_decrypteds], protocol=5)])]
            fit_ins = FitIns(parameters=fl.common.ndarrays_to_parameters(params_array), config=config_dict)
            return [(client_dict["swift"], fit_ins)]




    def aggregate_fit(
        self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures
    ) -> Tuple[Optional[Parameters], dict]:
        if (n_failures := len(failures)) > 0:
            raise Exception(f"Had {n_failures} failures in round {server_round}")
        if server_round == 1:
            for client, result in results:
                if client.cid != "swift":
                    self.client_banks_dict[client.cid] = (result.metrics["message"], result.metrics["nonce"])
        elif server_round == 2:
            for client, result in results:
                if client.cid != "swift":
                    result_ndarrays = fl.common.parameters_to_ndarrays(result.parameters)
                    self.client_to_pk[client.cid] = result_ndarrays[0]
                    self.client_to_okvs[client.cid] = result_ndarrays[1]
        elif server_round - 3 < self.num_bank_clients:
            pass # no messages being sent to aggregator here
        elif server_round - 3 == self.num_bank_clients:
            for client, result in results:
                if client.cid == "swift":
                    result_ndarrays = pickle.loads(fl.common.parameters_to_ndarrays(result.parameters)[0][0])
                    self.ciphertexts = result_ndarrays
        elif server_round - 4 == self.num_bank_clients:
            for client, result in results:
                if client != "swift":
                    result_ndarrays = pickle.loads(fl.common.parameters_to_ndarrays(result.parameters)[0][0])
                    self.randomized_ciphers[client.cid] = result_ndarrays
        elif server_round - 5 == self.num_bank_clients:
            for client, result in results:
                if client.cid == "swift":
                    result_ndarrays = pickle.loads(fl.common.parameters_to_ndarrays(result.parameters)[0][0])
                    self.ciphertexts = result_ndarrays
        elif server_round - 6 == self.num_bank_clients:
            for client, result in results:
                if client != "swift":
                    result_ndarrays = pickle.loads(fl.common.parameters_to_ndarrays(result.parameters)[0][0])
                    self.randomized_ciphers[client.cid] = result_ndarrays


        return None, {}

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not running any federated evaluation."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not aggregating any evaluation."""
        return None

    def evaluate(self, server_round: int, parameters: fl.common.typing.Parameters):
        """Not running any centralized evaluation."""
        return None
