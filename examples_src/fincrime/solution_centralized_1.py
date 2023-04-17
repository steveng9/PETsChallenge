from pathlib import Path

from loguru import logger
import pandas as pd
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import lightgbm as lgb

from datetime import datetime

SWIFT_COLS = ['SettlementAmount', 'InstructedAmount', 'hour',
              'sender_hour_freq', 'sender_currency_freq',
              'sender_currency_amount_average',
              'sender_receiver_freq',  # , 'beneficiary_account_num_transactions']
              'InterimTime']


def fit(swift_data_path: Path, bank_data_path: Path, model_dir: Path):

    # Load data
    logger.info("Loading data...")
    train = pd.read_csv(swift_data_path, index_col="MessageId")
    train_bank = pd.read_csv(bank_data_path, dtype=pd.StringDtype())
    # train_bank = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/dev_bank_dataset.csv", dtype=pd.StringDtype())
    # train = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/dev_swift_transaction_train_dataset.csv", index_col="MessageId")
    # test = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/swift_transaction_test_dataset.csv", index_col="MessageId")

    # _____________________________________________________________________________________
    # Prepare SWIFT data
    logger.info("Preparing features on SWIFT model... (1/4)")

    # TODO is this important? More efficient?
    # logger.info("Preparing train data...")
    # train = add_finalreceiver_col(train)
    # train = join_flags_to_swift_data(
    #     swift_df=train,
    #     bank_df=train_bank,
    # )

    train["Timestamp"] = train["Timestamp"].astype("datetime64[ns]")

    logger.info("Preparing features on SWIFT model... (2/4)")

    # Hour
    train["hour"] = train["Timestamp"].dt.hour

    # Hour frequency for each sender
    senders = train["Sender"].unique()
    train["sender_hour"] = train["Sender"] + train["hour"].astype(str)
    sender_hour_frequency = {}
    for s in senders:
        sender_rows = train[train["Sender"] == s]
        for h in range(24):
            sender_hour_frequency[s + str(h)] = len(sender_rows[sender_rows["hour"] == h])
    train["sender_hour_freq"] = train["sender_hour"].map(sender_hour_frequency)

    logger.info("Preparing features on SWIFT model... (3/4)")

    # Sender-Currency Frequency and Average Amount per Sender-Currency
    # TODO Had to remove test sender_currency from feature creation
    train["sender_currency"] = train["Sender"] + train["InstructedCurrency"]
    # test["sender_currency"] = test["Sender"] + test["InstructedCurrency"]
    sender_currencies = [sc for sc in set(list(train["sender_currency"].unique()))] # TODO rewrite line
    sender_currency_freq_avg = {sc: [0, 0] for sc in sender_currencies}
    for _, row in train.iterrows():
        freq_avg = sender_currency_freq_avg[row["sender_currency"]]
        freq_avg[0] += 1
        freq_avg[1] += row["InstructedAmount"]

    logger.info("Preparing features on SWIFT model... 3.5")

    sender_currency_freq = {k: fr_av[0] for k, fr_av in sender_currency_freq_avg.items()}
    sender_currency_avg = {k: (fr_av[1] / fr_av[0]) if fr_av[0] != 0 else 0 for k, fr_av in sender_currency_freq_avg.items()}

    train["sender_currency_freq"] = train["sender_currency"].map(sender_currency_freq)
    train["sender_currency_amount_average"] = train["sender_currency"].map(sender_currency_avg)

    logger.info("Preparing features on SWIFT model... (4/4)")

    # Sender-Receiver Frequency
    train["sender_receiver"] = train["Sender"] + train["Receiver"]
    # test["sender_receiver"] = test["Sender"] + test["Receiver"]
    sender_receiver_freq = {}
    for sr in set(list(train["sender_receiver"].unique())):
        sender_receiver_freq[sr] = len(train[train["sender_receiver"] == sr])
    train["sender_receiver_freq"] = train["sender_receiver"].map(sender_receiver_freq)

    train["SettlementDate"] = train["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    train["InterimTime"] = train[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)


    # TODO 'beneficiary_account_num_transactions' feature?

    # _____________________________________________________________________________________
    # Train SWIFT model
    logger.info("Preparing SWIFT model for train...")

    # TODO Can't scale test if train is separated from predict()
    scaler = StandardScaler()
    train[SWIFT_COLS] = scaler.fit_transform(train[SWIFT_COLS])

    Y_train = train["Label"].values
    X_train_SWIFT = train[SWIFT_COLS].values

    lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression'}
    lgb_classifier = lgb.LGBMClassifier(**lgb_params)
    lgb_classifier.fit(X_train_SWIFT, Y_train)
    logger.info("...done fitting")

    # Save the model for inference
    joblib.dump(lgb_classifier, model_dir / 'lgb_classifier.pkl')
    joblib.dump(scaler, model_dir / 'scaler.pkl')

    joblib.dump(sender_hour_frequency, model_dir / 'sender_hour_frequency.pkl')
    joblib.dump(sender_currency_freq, model_dir / 'sender_currency_freq.pkl')
    joblib.dump(sender_currency_avg, model_dir / 'sender_currency_avg.pkl')
    joblib.dump(sender_receiver_freq, model_dir / 'sender_receiver_freq.pkl')
    # lgb_classifier.booster_.save_model(model_dir / "lgb_classifier.joblib")





def predict(
    swift_data_path: Path,
    bank_data_path: Path,
    model_dir: Path,
    preds_format_path: Path,
    preds_dest_path: Path,
):
    # Load data
    logger.info("Loading data...")

    test = pd.read_csv(swift_data_path, index_col="MessageId")
    train_bank = pd.read_csv(bank_data_path, dtype=pd.StringDtype())
    # train_bank = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/dev_bank_dataset.csv", dtype=pd.StringDtype())
    # test = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/dev_swift_transaction_test_dataset.csv", index_col="MessageId")

    lgb_classifier = joblib.load(model_dir / 'lgb_classifier.pkl')
    scaler = joblib.load(model_dir / 'scaler.pkl')
    sender_hour_frequency = joblib.load(model_dir / 'sender_hour_frequency.pkl')
    sender_currency_freq = joblib.load(model_dir / 'sender_currency_freq.pkl')
    sender_currency_avg = joblib.load(model_dir / 'sender_currency_avg.pkl')
    sender_receiver_freq = joblib.load(model_dir / 'sender_receiver_freq.pkl')

    # _____________________________________________________________________________________
    # Prepare extra SWIFT features

    logger.info("Preparing features on SWIFT model... (1/4)")

    test["Timestamp"] = test["Timestamp"].astype("datetime64[ns]")

    # Hour
    test["hour"] = test["Timestamp"].dt.hour

    logger.info("Preparing features on SWIFT model... (2/4)")

    # Hour frequency for each sender
    senders = test["Sender"].unique()
    test["sender_hour"] = test["Sender"] + test["hour"].astype(str)
    test["sender_hour_freq"] = test["sender_hour"].map(sender_hour_frequency)

    logger.info("Preparing features on SWIFT model... (3/4)")

    # Sender-Currency Frequency and Average Amount per Sender-Currency
    test["sender_currency"] = test["Sender"] + test["InstructedCurrency"]
    # sender_currencies = [sc for sc in set(list(test["sender_currency"].unique()))] # TODO rewrite line
    # sender_currency_test_avg = {sc: 0 for sc in sender_currencies}
    # for _, row in test.iterrows():
    #     sc = row["sender_currency"]
    #     sender_currency_freq[sc] += 1
    #     sender_currency_test_avg[sc] += row["InstructedAmount"]
    test["sender_currency_freq"] = test["sender_currency"].map(sender_currency_freq)
    test["sender_currency_amount_average"] = test["sender_currency"].map(sender_currency_avg)

    logger.info("Preparing features on SWIFT model... (4/4)")

    # Sender-Receiver Frequency
    test["sender_receiver"] = test["Sender"] + test["Receiver"]
    test["sender_receiver_freq"] = test["sender_receiver"].map(sender_receiver_freq)

    test["SettlementDate"] = test["SettlementDate"].apply(lambda date: datetime.strptime(str(date), "%y%m%d")).astype("datetime64[ns]")
    test["InterimTime"] = test[["Timestamp", "SettlementDate"]].apply(lambda x: (x.SettlementDate - x.Timestamp).total_seconds(), axis=1)

    # _____________________________________________________________________________________
    # Extract features from bank
    bank_ids = set(train_bank['Bank'])

    # Cannot check with single bank because it is possible that an end-end transaction has
    # multiple individual transactions. Such rows will have multiple banks ids but same
    # Acctid and Name. Need to check individually.
    acct_ids = set(train_bank['Account'])

    # Dictionary to make search faster
    acct_flag_search = train_bank[['Account', 'Flags']].set_index('Account').T.to_dict('list')
    account_name_search = train_bank[['Account', 'Name']].set_index('Account').T.to_dict('list')

    test['bankSenderExists'] = test['Sender'].apply(lambda x: x in bank_ids)
    test['bankReceiverExists'] = test['Receiver'].apply(lambda x: x in bank_ids)

    test['isValidOrderingAcct'] = test['OrderingAccount'].apply(lambda x: x in acct_ids)
    test['isValidBeneficiaryAcct'] = test['BeneficiaryAccount'].apply(lambda x: x in acct_ids)

    logger.info("Preparing features on bank... (2/5)")

    # TODO combine these two blocks (to get rid of tracking 'OrderingFlag' and 'BeneficiaryFlag')
    # Get Flags from bank
    test['OrderingFlag'] = test['OrderingAccount'].apply(
        lambda x: acct_flag_search[x][0] if x in acct_flag_search.keys() else 0)  # lambda x: 1 if x in acct_flag_search.keys() else 0, BETTER? lambda x: x in acct_flag_search.keys(), BETTER YET (booleans instead of 0/1)?
    test['BeneficiaryFlag'] = test['BeneficiaryAccount'].apply(
        lambda x: acct_flag_search[x][0] if x in acct_flag_search.keys() else 0)

    # isAcctFlagged
    test['isOrderingAcctFlagged'] = test['OrderingFlag'].apply(lambda x: 0 if int(x) == 0 else 1)
    test['isBeneficiaryAcctFlagged'] = test['BeneficiaryFlag'].apply(lambda x: 0 if int(x) == 0 else 1)

    logger.info("Preparing features on bank... (3/5)")

    def check_name_orderingacct(x):
        # TODO check 'in acct_ids' instead?
        return x.OrderingAccount in account_name_search.keys() \
               and x.OrderingName == account_name_search[x.OrderingAccount][0] # TODO instead, see if IN list
    def check_name_benefacct(x):
        return x.BeneficiaryAccount in account_name_search.keys() \
               and x.BeneficiaryName == account_name_search[x.BeneficiaryAccount][0]
    test['isOrderingNameCorrect'] = test[['OrderingAccount', 'OrderingName']].apply(
        lambda x: check_name_orderingacct(x), axis=1)
    test['isBeneficiaryNameCorrect'] = test[['BeneficiaryAccount', 'BeneficiaryName']].apply(
        lambda x: check_name_benefacct(x), axis=1)

    logger.info("Preparing features on bank... (4/5)")
    def check_name_order(x):
        return (x.isValidOrderingAcct and pd.isna(x.OrderingName)) or x.isOrderingNameCorrect
    def check_name_benef(x):
        return (x.isValidBeneficiaryAcct and pd.isna(x.BeneficiaryName)) or x.isBeneficiaryNameCorrect
    test['isOrderingNameCorrect'] = test[['isValidOrderingAcct', 'OrderingName', 'isOrderingNameCorrect']].apply(
        lambda x: check_name_order(x), axis=1)
    test['isBeneficiaryNameCorrect'] = test[['isValidBeneficiaryAcct', 'BeneficiaryName', 'isBeneficiaryNameCorrect']].apply(
        lambda x: check_name_benef(x), axis=1)

    logger.info("Preparing features on bank... (5/5)")

    def get_valid(x):
        return x.isValidOrderingAcct and \
            x.isValidBeneficiaryAcct and \
            x.isOrderingNameCorrect and \
            x.isBeneficiaryNameCorrect

    test['isValidAcctInfo'] = test.apply(lambda x: get_valid(x), axis=1)

    # _____________________________________________________________________________________

    logger.info("Inference... (1/5)")

    test[SWIFT_COLS] = scaler.transform(test[SWIFT_COLS])

    # Y_test = test["Label"].values
    X_test_SWIFT = test[SWIFT_COLS].values

    # load the trained model
    pred_lgb = lgb_classifier.predict(X_test_SWIFT)
    pred_proba_lgb = lgb_classifier.predict_proba(X_test_SWIFT)[:, 1]

    logger.info("Inference... (2/5)")

    # print("AUPRC:", metrics.average_precision_score(y_true=Y_test, y_score=pred_proba_lgb))

    # _____________________________________________________________________________________

    ensemble_df = pd.DataFrame()
    ensemble_df['SWIFT'] = pred_proba_lgb

    ensemble_df['Rule_corridors'] = 1 - (test['bankSenderExists'] & test['bankReceiverExists']).values
    ensemble_df['ValidOR'] = 1 - test['isValidAcctInfo'].values
    ensemble_df['OrderFlag'] = test['isOrderingAcctFlagged'].values
    ensemble_df['BenefFlag'] = test['isBeneficiaryAcctFlagged'].values
    # ensemble_df['Label'] = test['Label'].values

    def getRule(x):
        if x.ValidOR == 1 or x.OrderFlag == 1 or x.BenefFlag == 1:
            return 1
        else:
            return 0

    ensemble_df['Rule_bank'] = ensemble_df.apply(lambda x: getRule(x), axis=1)
    ensemble_df['Rule+SWIFT'] = ensemble_df[['Rule_corridors', 'Rule_bank', 'SWIFT', ]].max(axis=1)

    # ensemble_df.to_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/ensemble_df.csv")

    # print("AUPRC:", metrics.average_precision_score(y_true=ensemble_df['Label'].values, y_score=ensemble_df['Rule+SWIFT'].values))

    ensemble_df['SWIFT_pred'] = pred_lgb
    ensemble_df['Rule+SWIFT_pred'] = ensemble_df[['Rule_corridors', 'SWIFT_pred', 'Rule_bank', ]].max(axis=1)
    # print("LGB Confusion Matrix=\n\n", confusion_matrix(Y_test, ensemble_df['Rule+SWIFT_pred'].values))

    logger.info("Inference... (5/5)")

    preds_format_df = pd.read_csv(preds_format_path, index_col="MessageId")
    preds_format_df = preds_format_df.assign(Score=ensemble_df['Rule+SWIFT'].values)

    logger.info("Writing out test predictions...")
    preds_format_df.to_csv(preds_dest_path)
    logger.info("Done.")
