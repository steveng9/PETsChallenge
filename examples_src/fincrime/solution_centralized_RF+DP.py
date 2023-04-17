from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import numpy as np
from sklearn import metrics

from forest import RandomForestClassifier

from sklearn.preprocessing import OneHotEncoder

from examples_src.fincrime.diffprivlib.quantiles import percentile
from examples_src.fincrime.diffprivlib.tools_utils import mean

DATA_DIR = "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/"
MODEL_DIR = "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/centralized/"


SWIFT_COLS = ['SameCurrency']
BIN_GRANULARITY = 100


def extract_swift_features(swift_df: pd.DataFrame):

    logger.info("Swift feature extraction : hour")
    # hour
    # swift_df["Timestamp"] = swift_df["Timestamp"].astype("datetime64[ns]")
    #
    # logger.info("Swift feature extraction: InterimTime")
    # swift_df["SettlementDate"] = swift_df["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    # swift_df["InterimTime"] = swift_df[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)
    #
    # logger.info("Swift feature extraction: Difference in days (abs)")
    # swift_df['SettlementDate'] = swift_df['SettlementDate'].astype("datetime64[ns]")
    # swift_df['difference_days_absolute'] = abs(swift_df['SettlementDate'].dt.day - pd.to_datetime(swift_df['Timestamp']).dt.day)
    #
    # logger.info("Swift feature extraction: SameCurrency")
    # swift_df['SameCurrency'] = (swift_df['SettlementCurrency'] == swift_df['InstructedCurrency'])



def create_bins(swift_df: pd.DataFrame):
    bins1 = BIN_GRANULARITY
    bins2 = BIN_GRANULARITY

    split = mean(swift_df[swift_df["Label"] == 0]["InterimTime"], epsilon=1)
    fuzzy1 = 1000
    fuzzy2 = 1000

    begin1, end1 = percentile(swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] > split]["InterimTime"], [0, 100], epsilon=1)
    begin1 -= fuzzy1
    end1 += fuzzy1

    begin2, end2 = percentile(swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] < split]["InterimTime"], [0, 100], epsilon=1)
    begin2 -= fuzzy2
    end2 += fuzzy2

    edges2 = np.arange(begin2, end2, (end2 - begin2) / (bins2 + 1)).tolist()
    edges1 = np.arange(begin1, end1, (end1 - begin1) / (bins1 + 1)).tolist()
    edges2.extend(edges1)
    all_edges = np.array(edges2)
    print(len(all_edges))

    return all_edges

def one_hot_encode_interimTime(swift_df: pd.DataFrame, all_edges):
    bins = np.digitize(swift_df["InterimTime"], all_edges)
    swift_df["bins"] = bins.tolist()
    onehot_encoder = OneHotEncoder(categories=[list(range(len(all_edges) + 1))])
    encoded_matrix = onehot_encoder.fit_transform(swift_df[['bins']])
    encoder_df = pd.DataFrame.sparse.from_spmatrix(encoded_matrix).astype(np.int0)
    swift_df.drop(columns=["bins"], inplace=True)
    return swift_df.join(encoder_df)



def fit(swift_data_path: Path, model_dir: Path, epsilon):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)

    extract_swift_features(swift_df)
    all_edges = create_bins(swift_df)
    swift_df = one_hot_encode_interimTime(swift_df, all_edges)

    logger.info("dp normalize data...")
    Y_train = swift_df["Label"].values
    X_train = swift_df[SWIFT_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

    logger.info("fit model for epsilon {}...", epsilon)

    rf_dp_model = RandomForestClassifier(epsilon=5.0, n_estimators=20, max_depth=10)
    rf_dp_model.fit(X_train, Y_train)

    joblib.dump(all_edges, model_dir / 'all_edges_rf.joblib')
    joblib.dump(rf_dp_model, model_dir / 'rf_dp_model.joblib')




def predict(
        swift_data_path: Path,
        bank_data_path: Path,
        model_dir: Path
):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)
    bank_df = pd.read_csv(bank_data_path)

    extract_swift_features(swift_df)
    all_edges = joblib.load(model_dir / "all_edges_rf.joblib")
    swift_df = one_hot_encode_interimTime(swift_df, all_edges)

    logger.info("Swift local inference")

    logger.info("model data...")
    rf_dp_model = joblib.load(model_dir / 'rf_dp_model.joblib')

    logger.info("DP Normalize data...")

    X_test = swift_df[SWIFT_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

    pred_proba = rf_dp_model.predict_proba(X_test)[:, 1]

    # ________________________________________________________________________

    logger.info("Starting bank validity check")

    bank_ids = set(bank_df['Bank'])

    acct_flag_search = bank_df[['Account', 'Flags']].set_index('Account').to_dict()['Flags']
    acct_name_search = bank_df[['Account', 'Name']].set_index('Account').to_dict()['Name']
    acct_address_search = bank_df[['Account', 'Street']].set_index('Account').to_dict()['Street']

    def check_with_bank_info(trns):
        order_acct = trns.OrderingAccount
        benef_acct = trns.BeneficiaryAccount

        bank_info_valid = \
            trns.Sender in bank_ids and \
            trns.Receiver in bank_ids and \
            int(acct_flag_search.get(order_acct, 1)) == 0 and \
            int(acct_flag_search.get(benef_acct, 1)) == 0 and \
            acct_name_search.get(order_acct) == trns.OrderingName and \
            acct_name_search.get(benef_acct) == trns.BeneficiaryName and \
            acct_address_search.get(order_acct) == trns.OrderingStreet and \
            acct_address_search.get(benef_acct) == trns.BeneficiaryStreet

        return 0 if bank_info_valid else 1

    # ________________________________________________________________________

    # General final predictions
    logger.info("taking maximum of swift predictions and bank check")
    ensemble_df = pd.DataFrame()
    ensemble_df['SWIFT'] = pred_proba
    print("here now...")
    # ensemble_df['Bank'] = swift_df.apply(check_with_bank_info, axis=1)
    print("there now...")
    # ensemble_df['SWIFT+Bank'] = ensemble_df[['Bank', 'SWIFT']].max(axis=1)

    print("AUPRC on DP-MLP SWIFT:", metrics.average_precision_score(y_true=swift_df["Label"].values, y_score=ensemble_df['SWIFT'].values))
    # print("AUPRC on DP-MLP SWIFT + Bank:", metrics.average_precision_score(y_true=swift_df["Label"].values, y_score=ensemble_df['SWIFT+Bank'].values))



# fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 100000)
# predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 5.0)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 1.0)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 0.5)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))