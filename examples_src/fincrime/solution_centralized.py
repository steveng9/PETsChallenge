from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from datetime import datetime
from sklearn import metrics

from sklearn.ensemble import RandomForestClassifier

SWIFT_COLS = ['InterimTime', 'InstructedAmount', 'SameCurrency', 'difference_days_absolute']


def extract_swift_features(swift_df: pd.DataFrame):

    logger.info("Swift feature extraction : hour")
    # hour
    swift_df["Timestamp"] = swift_df["Timestamp"].astype("datetime64[ns]")

    logger.info("Swift feature extraction: InterimTime")
    swift_df["SettlementDate"] = swift_df["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    swift_df["InterimTime"] = swift_df[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)

    logger.info("Swift feature extraction: Difference in days (abs)")
    swift_df['SettlementDate'] = swift_df['SettlementDate'].astype("datetime64[ns]")
    swift_df['difference_days_absolute'] = abs(swift_df['SettlementDate'].dt.day - pd.to_datetime(swift_df['Timestamp']).dt.day)

    logger.info("Swift feature extraction: SameCurrency")
    swift_df['SameCurrency'] = (swift_df['SettlementCurrency'] == swift_df['InstructedCurrency'])


def fit(swift_data_path: Path, bank_data_path: Path, model_dir: Path):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)

    extract_swift_features(swift_df)

    logger.info("Swift local training")
    Y_train = swift_df["Label"].values
    X_train = swift_df[SWIFT_COLS].values

    RF_clf = RandomForestClassifier(n_estimators=10, max_depth=10)
    RF_clf.fit(X_train, Y_train)

    joblib.dump(RF_clf, model_dir / 'RF_clf.joblib')



def predict(
        swift_data_path: Path,
        bank_data_path: Path,
        model_dir: Path,
        preds_format_path: Path,
        preds_dest_path: Path,
):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path)
    bank_df = pd.read_csv(bank_data_path)

    extract_swift_features(swift_df)

    logger.info("Swift local inference")
    X_test_SWIFT = swift_df[SWIFT_COLS].values
    RF_clf = joblib.load(model_dir / "RF_clf.joblib")
    pred_proba_rf = RF_clf.predict_proba(X_test_SWIFT)[:, 1]

    # ________________________________________________________________________

    logger.info("Starting bank validity check")

    bank_ids = set(bank_df['Bank'])

    acct_flag_search = bank_df[['Account', 'Flags']].set_index('Account').to_dict()['Flags']
    acct_name_search = bank_df[['Account', 'Name']].set_index('Account').to_dict()['Name']
    acct_address_search = bank_df[['Account', 'Street']].set_index('Account').to_dict()['Street']
    acct_CCZ_search = bank_df[['Account', 'CountryCityZip']].set_index('Account').to_dict()['CountryCityZip']

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
            acct_address_search.get(benef_acct) == trns.BeneficiaryStreet and \
            acct_CCZ_search.get(order_acct) == trns.OrderingCountryCityZip and \
            acct_CCZ_search.get(benef_acct) == trns.BeneficiaryCountryCityZip

        return 0 if bank_info_valid else 1

    # ________________________________________________________________________

    # General final predictions
    logger.info("taking maximum of swift predictions and bank check")
    ensemble_df = pd.DataFrame()
    ensemble_df['SWIFT'] = pred_proba_rf
    ensemble_df['Bank'] = swift_df.apply(check_with_bank_info, axis=1)
    ensemble_df['SWIFT+Bank'] = ensemble_df[['Bank', 'SWIFT']].max(axis=1)
    # print("AUPRC RF:", metrics.average_precision_score(y_true=swift_df["Label"].values, y_score=ensemble_df['SWIFT'].values))
    # print("AUPRC RF:", metrics.average_precision_score(y_true=swift_df["Label"].values, y_score=ensemble_df['SWIFT+Bank'].values))

    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df = preds_format_df.assign(Score=ensemble_df['SWIFT+Bank'].values)

    logger.info("Writing out swift_df predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")
