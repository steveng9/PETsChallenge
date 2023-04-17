from pathlib import Path
import pandas as pd
import joblib
from sklearn import metrics
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.naive_bayes import CategoricalNB
from xgboost import XGBClassifier
import lightgbm as lgb

from forest import RandomForestClassifier

SWIFT_COLS = ['InterimTime']

# SWIFT_COLS = ['InterimTime', 'InstructedAmount', 'SameCurrency']
# SWIFT_COLS = ['InterimTime', 'SameCurrency']

ensemble_df1 = pd.DataFrame()

ensemble_df1["hi"] = [False, True, False]
print("test3", ensemble_df1['hi'].sum())

# train
print("train")
train = pd.read_csv("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/train.csv")
# train['SameCurrency'] = train[["SettlementCurrency", "InstructedCurrency"]].apply(lambda x: x.SettlementCurrency == x.InstructedCurrency, axis=1)

print("loaded train data")
Y_train = train["Label"].values
X_train = train[SWIFT_COLS].values
print("fitting on train")
# rf_model_dp = RandomForestClassifier(epsilon=5.0, n_estimators=20, max_depth=10)
rf_model_dp = RandomForestClassifier(epsilon=5.0, bounds=(-4.15e+08, 2e+07), n_estimators=20, random_state=0,
                                     max_depth=10, classes=[0, 1])
rf_model_dp.fit(X_train, Y_train)

# test
print("test")
test = pd.read_csv("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/test.csv")
# test['SameCurrency'] = test[["SettlementCurrency", "InstructedCurrency"]].apply(lambda x: x.SettlementCurrency == x.InstructedCurrency, axis=1)

print("loaded test data")
Y_test = test["Label"].values
X_test = test[SWIFT_COLS].values

print("predicting probability on test")
pred_proba_rfdp = rf_model_dp.predict_proba(X_test)[:, 1]
print("AUPRC with SWIFT only LGB:", metrics.average_precision_score(y_true=Y_test, y_score=pred_proba_rfdp))




ensemble_df = pd.DataFrame()
ensemble_df['SWIFT'] = pred_proba_rfdp

sndr_rcvr_outputs = joblib.load(
    "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/outputs_from_okvs_protocol.joblib")
client_to_banks = joblib.load(
    "C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/client_to_banks.joblib")
#
#
test["Bank_Validity"] = None

for (sndr, rcvr), outputs in sndr_rcvr_outputs.items():
    sender_banks = client_to_banks[sndr]
    receiver_banks = client_to_banks[rcvr]
    test.loc[
        (test['Sender'].isin(sender_banks)) & (test['Receiver'].isin(receiver_banks)), 'Bank_Validity'
    ] = outputs
    print("sum:", sum(outputs), len(outputs))
    print(None in test['Bank_Validity'].values)


ensemble_df['Bank'] = test["Bank_Validity"].apply(lambda x: 0 if x else 1).values
ensemble_df['SWIFT+Bank'] = ensemble_df[['SWIFT', 'Bank']].max(axis=1)

print(ensemble_df['SWIFT'].sum(), ensemble_df['Bank'].sum(), ensemble_df['SWIFT+Bank'].sum())

print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['SWIFT'].values))
print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['Bank'].values))
print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['SWIFT+Bank'].values))
