import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.naive_bayes import CategoricalNB
from loguru import logger
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder

import tensorflow_privacy as tp
import tensorflow as tf
from keras.models import model_from_json
from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise
from keras.callbacks import EarlyStopping

from examples_src.fincrime.diffprivlib.quantiles import percentile
from examples_src.fincrime.diffprivlib.tools_utils import mean


PNS_COLS = ['SameCurrency']
# SCALER_PNS_COLS = ['InstructedAmount', 'difference_days_absolute']
# CLIPS = {'InterimTime': 4000000, 'InstructedAmount': 2 * 10 ** 12, 'difference_days_absolute': 60}
BIN_GRANULARITY = 40


class PNSModel:
    def __init__(self):
        self.scaler = None
        self.lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression'}
        self.mlp_model = None
        self.all_edges = None
        self.dp_means = {}

    def fit(self, train: pd.DataFrame):
        # extract_features(train)
        self.all_edges = create_bins(train)
        encoder_df = one_hot_encode_interimTime(train, self.all_edges)
        encoder_df["SameCurrency"] = np.asarray(train["SameCurrency"]).astype(np.int64)

        Y_train = train["Label"].values
        X_train = encoder_df[PNS_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

        self.mlp_model = fit_dp(X_train, Y_train, 5.0)

        return self

    def predict(self, test: pd.DataFrame):
        # extract_features(test)
        encoder_df = one_hot_encode_interimTime(test, self.all_edges)
        encoder_df["SameCurrency"] = np.asarray(test["SameCurrency"]).astype(np.int64)

        # normalize_DP_test(test, self.dp_means)

        X_test = encoder_df[PNS_COLS + list(range(BIN_GRANULARITY * 2 + 3))].values

        self.mlp_model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy', 'auc'])
        pred_proba = self.mlp_model.predict(X_test).flatten()

        return pred_proba

    def save(self, path):
        # joblib.dump(self.xgb_classifier, path / 'xgb_model.joblib')
        # joblib.dump(self.lgb_classifier, path / 'lgb_model.joblib')
        # joblib.dump(self.rf_model_dp, path / 'rf_model_dp.joblib')

        joblib.dump(self.dp_means, path / 'dp_means.joblib')
        # joblib.dump(self.scaler, path / 'scaler.joblib')

        self.mlp_model.save_weights(path / 'mlp_weights.h5')
        json = self.mlp_model.to_json()
        joblib.dump(json, path / 'mlp_model_json.joblib')
        joblib.dump(self.all_edges, path / 'all_edges.joblib')

    @classmethod
    def load(cls, path):
        inst = cls()
        # inst.xgb_classifier = joblib.load(path / 'xgb_model.joblib')
        # inst.lgb_classifier = joblib.load(path / 'lgb_model.joblib')
        # inst.rf_model_dp = joblib.load(path / 'rf_model_dp.joblib')

        json = joblib.load(path / 'mlp_model_json.joblib')
        inst.mlp_model = model_from_json(json)
        inst.mlp_model.load_weights(path / 'mlp_weights.h5')
        inst.all_edges = joblib.load(path / "all_edges.joblib")

        # inst.scaler = joblib.load(path / 'scaler.joblib')
        inst.dp_means = joblib.load(path / 'dp_means.joblib')
        return inst

#
# def normalize_DP_train(pns_df: pd.DataFrame):
#     n = pns_df.shape[0]
#     logger.info("clipping values for DP normalization!")
#     # choose clip values
#     e_num = 0.03
#     e_den = 0.003
#
#     for f in SCALER_PNS_COLS:
#         pns_df[f] = pns_df[f].apply(lambda x: x if x < CLIPS[f] else CLIPS[f])
#     logger.info("Done clipping.")
#
#     dp_means = {}
#     for f in SCALER_PNS_COLS:
#         dp_means[f] = (np.sum(pns_df[f]) + np.random.default_rng().laplace(CLIPS[f] / e_num)) / (n + np.random.default_rng().laplace(1 / e_den))
#         pns_df[f] = pns_df[f] / dp_means[f]
#
#     pns_df["SameCurrency"] = np.asarray(pns_df["SameCurrency"]).astype(np.int64)
#
#     return dp_means
#
# def normalize_DP_test(pns_df: pd.DataFrame, dp_means):
#     logger.info("clipping values for DP normalization!")
#     # choose clip values
#
#     for f in SCALER_PNS_COLS:
#         pns_df[f] = pns_df[f].apply(lambda x: x if x < CLIPS[f] else CLIPS[f])
#     logger.info("Done clipping.")
#
#     # use means constructed from train
#     for f in SCALER_PNS_COLS:
#         pns_df[f] = pns_df[f] / dp_means[f]
#
#     pns_df["SameCurrency"] = np.asarray(pns_df["SameCurrency"]).astype(np.int64)
#


def create_bins(pns_df: pd.DataFrame):
    bins1 = BIN_GRANULARITY
    bins2 = BIN_GRANULARITY

    min1_ = -(13 * 365 + 4) * 86400
    split = mean(pns_df[pns_df["Label"] == 0]["InterimTime"], epsilon=0.01, bounds=(min1_, 0))
    fuzzy1 = 10_000
    fuzzy2 = 10_000

    min2_ = -1 * 86400
    begin1, end1 = percentile(pns_df[pns_df["Label"] == 0][pns_df["InterimTime"] > split]["InterimTime"],[0, 100], bounds=(min2_, 0), epsilon=0.3)
    begin1 -= fuzzy1
    end1 += fuzzy1

    begin2, end2 = percentile(pns_df[pns_df["Label"] == 0][pns_df["InterimTime"] < split]["InterimTime"],[0, 100], bounds=(min1_, (min1_ - min2_)), epsilon=0.3)
    begin2 -= fuzzy2
    end2 += fuzzy2

    edges2 = np.arange(begin2, end2, (end2 - begin2) / (bins2 + 1)).tolist()
    edges1 = np.arange(begin1, end1, (end1 - begin1) / (bins1 + 1)).tolist()
    edges2.extend(edges1)
    all_edges = np.array(edges2)

    return all_edges


def one_hot_encode_interimTime(pns_df: pd.DataFrame, all_edges):
    bins = np.digitize(pns_df["InterimTime"], all_edges)
    pns_df["bins"] = bins.tolist()
    onehot_encoder = OneHotEncoder(categories=[list(range(len(all_edges) + 1))])
    encoded_matrix = onehot_encoder.fit_transform(pns_df[['bins']])
    encoder_df = pd.DataFrame.sparse.from_spmatrix(encoded_matrix).astype(np.int0)
    return encoder_df



def extract_features(pns_df: pd.DataFrame):

    logger.info("feature extraction : hour")
    # hour
    pns_df["Timestamp"] = pns_df["Timestamp"].astype("datetime64[ns]")

    logger.info("feature extraction: InterimTime")
    pns_df["SettlementDate"] = pns_df["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    pns_df["InterimTime"] = pns_df[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)

    # logger.info("feature extraction: Difference in days (abs)")
    # pns_df['SettlementDate'] = pns_df['SettlementDate'].astype("datetime64[ns]")
    # pns_df['difference_days_absolute'] = abs(pns_df['SettlementDate'].dt.day - pd.to_datetime(pns_df['Timestamp']).dt.day)

    logger.info("feature extraction: SameCurrency")
    pns_df['SameCurrency'] = (pns_df['SettlementCurrency'] == pns_df['InstructedCurrency'])


class LearningRateReducerCb(tf.keras.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        old_lr = self.model.optimizer.lr.read_value()
        new_lr = old_lr * 0.9
        print("\nEpoch: {}. Reducing Learning Rate from {} to {}".format(epoch, old_lr, new_lr))
        self.model.optimizer.lr.assign(new_lr)


def fit_dp(X_train, Y_train, epsilon):
    logger.info("feature scaling : Scaling")
    logger.info("local training : Train on scaled and selected features")

    ################DG-SGD################
    n_features = X_train.shape[1]
    n = X_train.shape[0]
    epochs = 30
    batch_size = 600
    learning_rate = 0.05
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

    early = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)

    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=0.0, axis=-1,
                                              reduction=tf.losses.Reduction.NONE, name='binary_crossentropy')
    logger.info("Training DP-SGD Model")

    tf.keras.utils.disable_interactive_logging()

    mlp_model = tf.keras.models.Sequential()
    mlp_model.add(tf.keras.layers.Dense(units=1, input_shape=(n_features,), activation='sigmoid'))
    mlp_model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.AUC(curve="PR")])

    hist = mlp_model.fit(X_train, Y_train, batch_size=batch_size, validation_split=0.2, epochs=epochs,shuffle=True,
                         callbacks=[LearningRateReducerCb(), early])
    ################DG-SGD################

    return mlp_model



class BankModel:
    def __init__(self):
        self.pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        missing_values=pd.NA, strategy="constant", fill_value="-1"
                    ),
                ),
                ("encoder", OrdinalEncoder()),
                ("model", CategoricalNB()),
            ]
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if len(self.pipeline.named_steps["model"].classes_) == 1:
            # Training data only had class 0
            return pd.Series([0.0] * X.shape[0], index=X.index)
        return pd.Series(self.pipeline.predict_proba(X)[:, 1], index=X.index)

    def save(self, path):
        joblib.dump(self.pipeline, path)

    @classmethod
    def load(cls, path):
        inst = cls()
        inst.pipeline = joblib.load(path)
        return inst
