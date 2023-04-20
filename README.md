# PETsChallenge

Privacy-Preserving Feature Extraction for Detection of
Anomalous Financial Transactions

------------------------------------------------------------------------

This repository holds the code written by the PPMLHuskies for the 2nd Place solution in [the PETs Prize Challenge, Track A](https://www.drivendata.org/competitions/group/nist-federated-learning/). 
The technical report for our solution can be found [here](). 

# Description

The task is to predict probabilities for anomalous transactions, from a
synthetic database of international transactions, and several synthetic
databases of banking account information. We provide two solutions. One
solution, our centralized approach, found in `solution_centralized.py`,
uses the transactions database (PNS) and the banking database with no
privacy protections. The second solution, which provides robust privacy
gurantees outlined in our report, follows a federated architecture,
found in `solution_federated.py` and model.py. In this approach, PNS
data resides in one client, banking data is divided up accross other
clients, and an aggregator handles all the communication between any
clients. We have built in privacy protections so that clients and the
aggregator learn minimal information about each other, while engaging in
communication to detect anomalous transactions in PNS.

The way in which we conduct training and inference in both the
centralized and the federated architectures is fundamentally the same
(other than the privacy protections in the latter). Several new features
are engineered from the given PNS data. Then a model is trained on those
features from PNS. Next, during inference, a check is made to determine
if attributes from a PNS transaction match with the banking data, or if
the associated account in the banking data is flagged. If any of these
attributes are amiss, we give it a value of 1, and a 0 otherwise.
Lastly, we take the maximum of the inferred probabilities from the PNS
model, and the result from the Banking data validation, which is used as
our final prediction for the probability that the transaction is
anomalous.

The difference between the federated and centralized logic is that in
the federated set up, where there are one or multiple partitions of the
banking data across clients, is that the PNS client engages in a
cryptographic protocol based on homomorphic encryption with the banking
clients, routed through the aggregator, to perform feature extraction.
This protocol, to ensure privacy, and that PNS does not learn anything
from the banks beyond the set membership of a select few features, is
carried out over several rounds, r. r = 7 + n, where n is the number of
bank clients.

# Libraries and third party code used

-   tensorflow-privacy : for Differential Privacy optimizer in DP-SGD
    model

-   tensorflow (Keras) : for DP-SGD model in the federated solution.

-   numpy

-   pandas

-   pickle : to convert encrypted messages into a format compatable with
    flwr

-   joblib : to save components of our model within a client between
    fit() and predict(), and between rounds, where state is not saved.

-   sklearn : for our centralized model RandomForestClassifier

-   cryptography : for symmetric encryption of messages passed between
    clients through the aggregator

-   diffprivlib : the implementations from this package are used to
    compute feature boundaries using differential privacy

-   federated_fraud_detection : this is the package we've developed that
    allows for a multi-round homomorphic encryption protocol to carry
    out private feature extraction between the PNS client and the banks.

**Low-level cryptographic protocols in `federated_fraud_detection`**\
Our cryptographic protocols are implemented in pure Rust. The library is
designed to be highly-portable yet efficient. Internally, it relies on
the `okvs` crate to realize the OKVS. We ship a pre-built MIT-licensed
Python wrapper around our protocols called `federated_fraud_detection`,
which implements the following functionality:

-   `key_gen`: Generates a key pair for this party (PNS or a node),
    returning a tuple with (public_key, secret_key). The public_key must
    be sent to PNS.

-   `build_okvs`: Builds an OKVS. A node may encode multiple banks' data
    in a single OKVS. This function takes the public key of this node
    and the data from the database, which is a list of rows. Each row is
    a list of the relevant columns as a str. The OKVS must be sent to
    PNS.

-   `initiate_queries`: Called by PNS, this function takes the sender
    and receiver data for multiple queries at the same time. These
    inputs should have the same shape as when calling the build_okvs
    function. This function also takes the sender's and receiver's OKVS.
    The final argument is the public key of PNS. The returned object is
    a set of ciphertexts representing a batch of queries as a bytes
    object. This must be sent to the sending and receiving nodes.

-   `randomize`: This function is called by the sending and receiving
    nodes. The only argument is the set of ciphertexts that were
    generated by the initiate_queries function. The output is another
    bytes object representing randomized ciphertexts, which must be sent
    to PNS.

-   `combine`: This function for PNS inputs the randomized ciphertexts
    from the sending and receiving nodes (the order is actually not
    important) and outputs four sets of points as a tuple of bytes.
    **The order of these four sets of points is important!** Given
    output (a, b, c, d), PNS must send a to the sending node and b to
    the receiving node. It must store c and d locally.

-   `decrypt`: This function is called by the sending or receiving node.
    It inputs the points (bytes) from the combine function as well as
    the secret key of this node (also bytes). It outputs a set of
    'decrypted points', again as a bytes object. This object must be
    sent to PNS.

-   `finish`: This function is called by PNS, and it directly returns
    the feature for each queried transaction as a Boolean. The order of
    the inputs is crucial: first the sender's points from the decrypt
    function, then the receiver's, then c and d from the combine
    function, and finally the secret key of PNS as a bytes object. The
    result is a list of bools. The output should be correct with
    extremely high probability.

Instead of the `bytes` type, the most recent version of our wrapper
returns lists of byte-sized unsigned integers.

# Centralized solution

The centralized solution comprises of a fit(), and predict(). During
both, we engineer features from the test and train PNS dataset. During
fit(), we train a random forest model on a subset of these features,
`PNS_COLS`, and then save the model. During predict(), we load the
model, perform probability inferences on the PNS test data, and then
check each transaction against the bank data. We look for existence of
the account, a mismatch in the ordering or beneficiary name, and if
there are any flags on the account. If any of these are true, then we
make the label on the PNS for each account, `’invalid_details’`, equal
to 1, and 0 if nothing is amiss. Then we take the maximum of that
feature, and the probabilities computed from the model, and this becomes
our final classification.

## Structure of the source code

-   solution_centralized.py : holds all of the fit() and predict() logic

    We train a simple RandomForestClassifier on a select set of features
    from the PNS dataset:

            X_test_PNS = pns_df[PNS_COLS].values
            pred_proba_rf = RF_clf.predict_proba(X_test_PNS)[:,1]

    Here is where we begin extracting features from the bank:

            pns_df['bankSenderExists'] = pns_df['Sender'].apply(lambda x : x in bank_ids)
            pns_df['bankReceiverExists'] = pns_df['Receiver'].apply(lambda x : x in bank_ids)

    And this is where we ultimately compute the maximum of the
    prediction from the RF model and the invalidity boolean from the
    banking data:

            ensemble_df['PNS+BANK_RF'] = ensemble_df[['BANK','PNS_RF']].max(axis=1)

# Federated solution

For the federated code submission, we carry out a federated homomorphic
encryption communication protocol in order for PNS to discern private
set intersection with the banking data, which are partitioned across
several clients. This protocol prevents the aggregator from learning any
information from the banks' datasets, and prevents each banking client
from learning anything about each other, or PNS. PNS only learns 1)
which banks reside in which nodes (this information is encrypted so the
aggregator cannot know) and 2) whether the following set of fields of a
PNS transaction exactly match the fields on the corresponding bank
account entry. After checking if a transaction's bank exists, PNS learns
only whether - Account, OrderingName, OrderingStreet,
OrderingCountryCityZip all match with one of the sending bank's entry's
\[Account, Name, Street, Country\], or not, and whether - Account,
BeneficiaryName, BeneficiaryStreet, BeneficiaryCountryCityZip all match
with on of the beneficiary bank's entry's \[Account, Name, Street,
Country\], or not. If any of these fields do not match, then PNS does
*not* learn which one.

This protocol is carried out through nine steps:

1.  During setup, a pair of symmetric keys are pre-distributed to each
    client, and to PNS. This enables each client to encrypt which banks
    it has, and send this information to PNS in the first round. PNS
    then uses its pre-distributed key associated with the client who
    sent the message to decrypt the mapping.

            for (client, banks_encrypted, nonce) in zip(*parameters):
            key = joblib.load(self.client_dir / (client + '_symmetric_key.joblib'))
            client_to_banks[client] = decrypt_banks(banks_encrypted, nonce, key)

2.  Each client creates its own oblivious value key store (OKVS),
    containing all of its unflagged transactions, and creates its own
    pair of keys (one private, one public).

            pk, sk = key_gen()
            self.bank_df['Flags'] = self.bank_df['Flags'].apply(lambda x: int(x))
            okvs = build_okvs(pk, self.bank_df.loc[self.bank_df['Flags'] == 0][
                ["Account", "Name", 'Street', 'CountryCityZip']
            ].values.astype("U"))

    It sends the OKVS to PNS, as well as its public key.

3.  Meanwhile, PNS determines all sending-receiving pairs of clients,
    based on its transaction data, and its mapping of which clients
    store which banks.

            all_sending_receiving_pairs_clients = ...

4.  The aggregator will send the OKVSs and the clients' public keys (one
    per round due to their heft) to PNS:

            elif server_round >= 3 and server_round - 3 < self.num_bank_clients:
                config_dict["okvs protocol"] = "step1"
                clients = sorted(client_dict.keys())
                client_chosen = clients[server_round-3]
                pk = self.client_to_pk[client_chosen]
                okvs = self.client_to_okvs[client_chosen]

5.  For every sending - receiving client pair, PNS generates a
    ciphertext of the sender information, and a ciphertext of the
    receiver information, to be sent to the sending client and the
    receiving client for further processing. And for each of these
    pairings, PNS keeps a public key, and a private key `sk` to use at
    the end of the protocol:

            ciphertext = initiate_queries(
                sender_data, 
                receiver_data,
                sender_okvs.tolist(),
                receiver_okvs.tolist(),
                pk
            )

6.  In this step, each banking client will receive a set of its sending
    ciphertexts, and its receiving ciphertexts (one for each other
    client its account holders transact with). The client carries out a
    'randomize' computation on the ciphertext, and then sends it back to
    PNS.

            sender_ciphers_randomized = [
                np.array(randomize(ciphertext.tolist()), dtype=np.uint8)
                    for ciphertext in sender_ciphers
            ]
            receiver_ciphers_randomized = [
                np.array(randomize(ciphertext.tolist()), dtype=np.uint8)
                    for ciphertext in receiver_ciphers
            ]

7.  Then PNS, upon receiving all of the randomizations from banking
    clients, combines the corresponding sending and receiving
    randomizations derived from the ciphertexts it sent out. And for
    each of the combinations of sender / receiver randomized
    ciphertexts, PNS sends the generated a to the sender, b to the
    receiver, and stores c, and d for later computation.

            a, b, c, d = combine(sndr_randomized.tolist(), rcvr_randomized.tolist())
            sndr_rcvr_c_d[(sndr, rcvr)] = (c, d)
            // send a and b off to sender and receiver clients

8.  The banking clients, in the same coordinated fashion as step 6,
    perform the penultimate decryption, using their received `a`s and
    `b`s from PNS.

            sender_decrypteds = [
                np.array(decrypt(a.tolist(), secret_key), dtype=np.uint8) for a in As
            ]
            receiver_decrypteds = [
                np.array(decrypt(b.tolist(), secret_key), dtype=np.uint8) for b in Bs
            ]

9.  Lastly, PNS computes the private set intersection of the
    transactions it originally sent as ciphertexts to the sender /
    receiver client pair, by using its unique private for that pairing,
    and its stored c and d values:

            outputs = finish(
                sndr_decrypted.tolist(),
                rcvr_decrypted.tolist(),
                c,
                d,
                secret_key_for_pair
            )

    These outputs are simply booleans, of whether or not the entire set
    of \[Account, OrderingName, OrderingStreet, OrderingCountryCityZip\]
    match with those of the sender bank AND the entire set of \[Account,
    BeneficiaryName, BeneficiaryStreet, BeneficiaryCountryCityZip\]
    match those of the beneficiary bank AND neither bank accounts are
    flagged.

    This supplemental information on each transaction is interpreted as
    1 if the private set intersection failed, and 0 otherwise. Then, for
    our final probability determination, PNS takes the maximum of this
    value, and a differentially private model's predicted probability.

            ensemble_df['Bank'] = self.pns_df["Bank_Validity"].apply(
                lambda x: 0 if x else 1
            ).values
            ensemble_df['PNS+Bank'] = ensemble_df[['PNS', 'Bank']].max(axis=1)

When training our model on the PNS data, we make all features that are
engineered on knowledge of the entire dataset differentially private by
using functions, such as `mean()` and `predict()` from `diffprivlib`.

The model we train is itself differentially private. We use DP-SGD from
`tensorflow_privacy`.

## Structure of the source code

-   `solution_federated.py` : handles all of the homomorphic encryption
    communication logic between aggregator, PNS, and Banking clients

-   `model.py` : contains the PNS feature engineering, the training
    logic, and the inference logic

-   `federated_fraud_detection...whl` : contains the vendored OKVS
    homomorphic encryption functions for our private set intersection
    communication

-   `install.sh` : for installing the `federated_fraud_detection`
    package.

-   `base.py` : implementation from `diffprivlib` for making the feature
    engineering process differentially private

-   `binary.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `exponential.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `geometric.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `laplace.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `quantiles.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `tools_utils.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `utils.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private

-   `validation.py` : implementation from `diffprivlib` for making the
    feature engineering process differentially private
