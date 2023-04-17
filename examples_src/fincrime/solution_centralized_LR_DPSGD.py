from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
import numpy as np
from sklearn import metrics

import tensorflow_privacy as tp
import tensorflow as tf
from keras.models import model_from_json
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

from sklearn.preprocessing import OneHotEncoder

from examples_src.fincrime.diffprivlib.quantiles import percentile
from examples_src.fincrime.diffprivlib.tools_utils import mean

DATA_DIR = "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/"
MODEL_DIR = "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/centralized/"


SWIFT_COLS = ['SameCurrency']
SCALER_COLS = ['InstructedAmount', 'difference_days_absolute']
CLIPS = {'InterimTime': 4000000, 'InstructedAmount': 2 * 10 ** 12, 'difference_days_absolute': 60}
BIN_GRANULARITY = 100

def normalize_DP_train(swift_df: pd.DataFrame):
    n = swift_df.shape[0]
    logger.info("clipping values for DP normalization!")
    # choose clip values
    e_num = 0.03
    e_den = 0.003

    for f in SCALER_COLS:
        swift_df[f] = swift_df[f].apply(lambda x: x if x < CLIPS[f] else CLIPS[f])
    logger.info("Done clipping.")

    dp_means = {}
    for f in SCALER_COLS:
        dp_means[f] = (np.sum(swift_df[f]) + np.random.default_rng().laplace(CLIPS[f] / e_num)) / (n + np.random.default_rng().laplace(1 / e_den))
        swift_df[f] = swift_df[f] / dp_means[f]

    swift_df["SameCurrency"] = np.asarray(swift_df["SameCurrency"]).astype(np.int64)

    return dp_means

def normalize_DP_test(swift_df: pd.DataFrame, dp_means):
    logger.info("clipping values for DP normalization!")
    # choose clip values

    for f in SCALER_COLS:
        swift_df[f] = swift_df[f].apply(lambda x: x if x < CLIPS[f] else CLIPS[f])
    logger.info("Done clipping.")

    # use means constructed from train
    for f in SCALER_COLS:
        swift_df[f] = swift_df[f] / dp_means[f]

    swift_df["SameCurrency"] = np.asarray(swift_df["SameCurrency"]).astype(np.int64)



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

    benign_min = swift_df[swift_df["Label"] == 0]["InterimTime"].min()
    benign_max = swift_df[swift_df["Label"] == 0]["InterimTime"].max()

    split = mean(swift_df[swift_df["Label"] == 0]["InterimTime"], bounds=(benign_min, benign_max), epsilon=0.01)
    fuzzy1 = 10_000
    fuzzy2 = 10_000

    upper_min = swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] > split]["InterimTime"].min()
    begin1, end1 = percentile(swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] > split]["InterimTime"], [0, 100], bounds=(upper_min, benign_max), epsilon=1)
    begin1 -= fuzzy1
    end1 += fuzzy1

    lower_max = swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] < split]["InterimTime"].max()
    begin2, end2 = percentile(swift_df[swift_df["Label"] == 0][swift_df["InterimTime"] < split]["InterimTime"], [0, 100], bounds=(benign_min, lower_max), epsilon=1)
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



def fit_dp(X_train, Y_train, epsilon):
    logger.info("Swift feature scaling : Scaling")
    logger.info("Swift local training : Train on scaled and selected features")

    ################DG-SGD################
    n_features = X_train.shape[1]
    n = X_train.shape[0]
    epochs = 20
    batch_size = 600
    learning_rate = 0.03
    num_microbatches = 1
    l2_norm_clip = 0.001
    delta = 1.0 / n
    noise_lbd = 0.01

    noise_multiplier = compute_noise(n=n, batch_size=batch_size, target_epsilon=epsilon, epochs=epochs, delta=delta,
                                     noise_lbd=noise_lbd)

    optimizer = tp.DPKerasAdamOptimizer(l2_norm_clip=l2_norm_clip,
                                        noise_multiplier=noise_multiplier,
                                        num_microbatches=num_microbatches,
                                        learning_rate=learning_rate)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1,
                                              reduction=tf.losses.Reduction.NONE, name='binary_crossentropy')
    logger.info("Training DP-SGD Model")

    # tf.keras.utils.disable_interactive_logging()

    mlp_model = tf.keras.models.Sequential()
    mlp_model.add(tf.keras.layers.Dense(units=1, input_shape=(n_features,), activation='sigmoid'))
    mlp_model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.AUC(curve="PR")])

    hist = mlp_model.fit(X_train, Y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs,
                         shuffle=True)
    ################DG-SGD################

    return mlp_model


def fit(swift_data_path: Path, model_dir: Path, epsilon):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)

    extract_swift_features(swift_df)
    all_edges = create_bins(swift_df)
    swift_df = one_hot_encode_interimTime(swift_df, all_edges)

    # dp_means = normalize_DP_train(swift_df)

    Y_train = swift_df["Label"].values
    X_train = swift_df[SWIFT_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

    logger.info("fit model for epsilon {}...", epsilon)
    mlp_model = fit_dp(X_train, Y_train, epsilon)

    # joblib.dump(dp_means, model_dir / 'dp_means.joblib')
    joblib.dump(all_edges, model_dir / 'all_edges.joblib')
    mlp_model.save_weights(model_dir / 'mlp_weights.h5')
    json = mlp_model.to_json()
    joblib.dump(json, model_dir / 'mlp_model_json.joblib')




def predict(
        swift_data_path: Path,
        bank_data_path: Path,
        model_dir: Path
):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)
    bank_df = pd.read_csv(bank_data_path)

    extract_swift_features(swift_df)
    all_edges = joblib.load(model_dir / "all_edges.joblib")
    swift_df = one_hot_encode_interimTime(swift_df, all_edges)

    logger.info("Swift local inference")

    logger.info("model data...")
    json = joblib.load(model_dir / 'mlp_model_json.joblib')
    mlp_model = model_from_json(json)
    mlp_model.load_weights(model_dir / 'mlp_weights.h5')
    # dp_means = joblib.load(model_dir / 'dp_means.joblib')

    # normalize_DP_test(swift_df, dp_means)

    X_test = swift_df[SWIFT_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

    logger.info("predict MLP...")
    mlp_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', 'auc'])
    pred_proba = mlp_model.predict(X_test).flatten()

    del mlp_model

    # ________________________________________________________________________

    logger.info("Starting bank validity check")

    bank_ids = set(bank_df['Bank'])

    acct_flag_search = bank_df[['Account', 'Flags']].set_index('Account').to_dict()['Flags']
    acct_name_search = bank_df[['Account', 'Name']].set_index('Account').to_dict()['Name']
    acct_address_search = bank_df[['Account', 'Street']].set_index('Account').to_dict()['Street']

    count = 0

    def check_with_bank_info(trns):
        order_acct = trns.OrderingAccount
        benef_acct = trns.BeneficiaryAccount
        nonlocal count
        if count % 100 == 0:
            print(count)
        count += 1

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



fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 100000)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 5.0)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 1.0)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))

fit(Path(DATA_DIR + "train.csv"), Path(MODEL_DIR), 0.5)
predict(Path(DATA_DIR + "test.csv"), Path(DATA_DIR + "dev_bank_dataset.csv"), Path(MODEL_DIR))