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


SWIFT_COLS = ['SettlementAmount', 'InstructedAmount', 'hour',
              'sender_hour_freq', 'sender_currency_freq',
              'sender_currency_amount_average', 'sender_receiver_freq',
              'InterimTime']

#
# # train
# print("train")
# train = pd.read_csv("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/train.csv")
# print("loaded train data")
# scaler = StandardScaler()
# lgb_params = {'boosting_type': 'gbdt', 'objective': 'regression'}
# lgb_classifier = lgb.LGBMClassifier(**lgb_params)
# # self.xgb_classifier = XGBClassifier(n_estimators=100,random_state=0)
# train[SWIFT_COLS] = scaler.fit_transform(train[SWIFT_COLS])
# Y_train = train["Label"].values
# X_train = train[SWIFT_COLS].values
# print("fitting on train")
# lgb_classifier.fit(X_train, Y_train)
#
#
#
# # test
# print("test")
# test = pd.read_csv("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/test.csv")
# print("loaded test data")
# test[SWIFT_COLS] = scaler.transform(test[SWIFT_COLS])
# Y_test = test["Label"].values
# X_test = test[SWIFT_COLS].values
#
# print("predicting probability on test")
# # pred_proba_xgb = self.xgb_classifier.predict_proba(X_test)[:, 1]
# pred_proba_lgb = lgb_classifier.predict_proba(X_test)[:, 1]
# # print("AUPRC with SWIFT only XGB:", metrics.average_precision_score(y_true=Y_test, y_score=pred_proba_xgb))
# print("AUPRC with SWIFT only LGB:", metrics.average_precision_score(y_true=Y_test, y_score=pred_proba_lgb))
#
#
#
# # bank extraction
# print()
# print("bank extraction")
# ensemble_df = pd.DataFrame()
# ensemble_df['SWIFT'] = pred_proba_lgb
#
# joblib.dump(ensemble_df, Path("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/ensemble_df_test.joblib"))


print("loading 1")
test = pd.read_csv("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/test.csv")
print("loading 2")
ensemble_df = joblib.load(Path("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/ensemble_df_test.joblib"))
print("loading 3")
sndr_rcvr_outputs = joblib.load("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/outputs_from_okvs_protocol.joblib")
client_to_banks = joblib.load("C:/Users/deek/PycharmProjects/pythonProject-PETs3/data/fincrime/state/scenario01/swift/client_to_banks.joblib")


test["Bank_Validity"] = False

for (sndr, rcvr), outputs in sndr_rcvr_outputs.items():
    sender_banks = client_to_banks[sndr]
    receiver_banks = client_to_banks[rcvr]
    test.loc[
        (test['Sender'].isin(sender_banks)) & (test['Receiver'].isin(receiver_banks)), 'Bank_Validity'
    ] = outputs
    print(None in test['Bank_Validity'].values)

ensemble_df['Bank'] = test["Bank_Validity"].apply(lambda x: 0 if x else 1)
print(ensemble_df["Bank"].sum())

ensemble_df['SWIFT+Bank'] = ensemble_df[['SWIFT', 'Bank']].max(axis=1)

print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['SWIFT'].values))
print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['Bank'].values))
print("AUPRC LGB:", metrics.average_precision_score(y_true=test["Label"].values, y_score=ensemble_df['SWIFT+Bank'].values))



