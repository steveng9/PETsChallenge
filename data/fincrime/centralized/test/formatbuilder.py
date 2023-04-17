import pandas as pd

swift_df = pd.read_csv("C:/Users/deek/Documents/School/PET Prize/pets-prize-challenge-runtime/data/fincrime/dev_swift_transaction_test_dataset.csv")

format_df = pd.DataFrame()
format_df['MessageId'] = swift_df['MessageId']
format_df['Score'] = 0.5
print(format_df)
format_df.to_csv('predictions_format.csv', index=False)