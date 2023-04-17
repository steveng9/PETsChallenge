from pathlib import Path

import joblib
from loguru import logger
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import ShuffleSplit, StratifiedKFold, StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import sklearn.utils

from xgboost import XGBClassifier
import lightgbm as lgb
# DATA_DIR = "/Users/sikha/Documents/"
DATA_DIR = "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/"

def fit(swift_data_path: Path):

    logger.info("Loading data...")
    swift_df = pd.read_csv(swift_data_path / "dev_swift_transaction_train_dataset.csv")

    logger.info("Timestamp")
    swift_df["Timestamp"] = swift_df["Timestamp"].astype("datetime64[ns]")

    logger.info("Swift feature extraction: InterimTime")
    swift_df["SettlementDate"] = swift_df["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    swift_df["InterimTime"] = swift_df[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)

    logger.info("Swift feature extraction: Difference in days (abs)")
    swift_df['SettlementDate'] = swift_df['SettlementDate'].astype("datetime64[ns]")
    swift_df['difference_days_absolute'] = abs(swift_df['SettlementDate'].dt.day - pd.to_datetime(swift_df['Timestamp']).dt.day)

    logger.info("Swift feature extraction: SameCurrency")
    swift_df['SameCurrency'] = (swift_df['SettlementCurrency'] == swift_df['InstructedCurrency'])
    #
    # logger.info("Swift feature extraction : hour frequency for each sender")
    # # hour frequency for each sender
    # senders = swift_df["Sender"].unique()
    # swift_df["sender_hour"] = swift_df["Sender"] + swift_df["hour"].astype(str)
    # sender_hour_frequency = {}
    # for s in senders:
    #     sender_rows = swift_df[swift_df["Sender"] == s]
    #     for h in range(24):
    #         sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])
    # swift_df["sender_hour_freq"] = swift_df["sender_hour"].map(sender_hour_frequency)
    #
    # logger.info("Swift feature extraction : Sender-Currency Frequency and Average Amount per Sender-Currency")
    # #Sender-Currency Frequency and Average Amount per Sender-Currency
    # swift_df["sender_currency"] = swift_df["Sender"] + swift_df["InstructedCurrency"]
    #
    # ##### Optimized
    # ## start
    # sender_currencies = [sc for sc in set(list(swift_df["sender_currency"].unique()))]
    # sender_currency_freq_avg = {sc: [0,0] for sc in sender_currencies}
    # # sender_currency_avg = {sc: 0 for sc in sender_currencies}
    # for _, row in swift_df.iterrows():
    #     freq_avg = sender_currency_freq_avg[row["sender_currency"]]
    #     freq_avg[0] += 1
    #     freq_avg[1] += row["InstructedAmount"]
    #
    # #logger.info("Preparing features on SWIFT model... 3.5")
    #
    # sender_currency_freq = {k: fr_av[0] for k, fr_av in sender_currency_freq_avg.items()}
    # sender_currency_avg = {k: (fr_av[1] / fr_av[0]) if fr_av[0] != 0 else 0 for k, fr_av in sender_currency_freq_avg.items()}
    #
    # ## end
    #
    # swift_df["sender_currency_freq"] = swift_df["sender_currency"].map(sender_currency_freq)
    # swift_df["sender_currency_amount_average"] = swift_df["sender_currency"].map(
    #     sender_currency_avg
    # )
    #
    # logger.info("Swift feature extraction : Sender-Receiver Frequency")
    # #Sender-Receiver Frequency
    # swift_df["sender_receiver"] = swift_df["Sender"] + swift_df["Receiver"]
    # sender_receiver_freq = {}
    #
    # for sr in set(
    #         list(swift_df["sender_receiver"].unique())):
    #     sender_receiver_freq[sr] = len(swift_df[swift_df["sender_receiver"] == sr])
    #
    # swift_df["sender_receiver_freq"] = swift_df["sender_receiver"].map(sender_receiver_freq)



    logger.info("Loading data...")
    test_df = pd.read_csv(swift_data_path / "dev_swift_transaction_test_dataset.csv")

    logger.info("Timestamp")
    test_df["Timestamp"] = test_df["Timestamp"].astype("datetime64[ns]")

    logger.info("Swift feature extraction: InterimTime")
    test_df["SettlementDate"] = test_df["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    test_df["InterimTime"] = test_df[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)

    logger.info("Swift feature extraction: Difference in days (abs)")
    test_df['SettlementDate'] = test_df['SettlementDate'].astype("datetime64[ns]")
    test_df['difference_days_absolute'] = abs(test_df['SettlementDate'].dt.day - pd.to_datetime(test_df['Timestamp']).dt.day)

    logger.info("Swift feature extraction: SameCurrency")
    test_df['SameCurrency'] = (test_df['SettlementCurrency'] == test_df['InstructedCurrency'])



    swift_df.to_csv(Path(DATA_DIR + "train.csv"))
    test_df.to_csv(Path(DATA_DIR + "test.csv"))




fit(Path(DATA_DIR))
