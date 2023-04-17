from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict


# Define write paths for partitions
d = Path("/")
OUT_DIR = d / "data/fincrime/scenario01"
TEST = OUT_DIR / "test"
TRAIN = OUT_DIR / "train"

# copy train dataset to `swift` partitions
(TEST / "swift").mkdir(exist_ok=True, parents=True)
(TRAIN / "swift").mkdir(exist_ok=True, parents=True)



test_df = pd.read_csv(d / "data/fincrime/dev_swift_transaction_test_dataset.csv", index_col="MessageId")
label_col = "Label"

# create pred format df and set all predictions to 0.5
test_df["Score"] = 0.5
pred_format_df = test_df[["Score"]].copy()

# create test df from remaining columns
test_df = test_df.drop(columns=[label_col, "Score"])



# write out to `swift` partition
test_path = d / "data/fincrime/dev_swift_transaction_test_dataset.csv"
pred_path = d / "data/fincrime/scenario01/test/swift/predictions_format.csv"

# test_df.to_csv(test_path)
pred_format_df.to_csv(pred_path)


bank_df = pd.read_csv(d / "data/fincrime/dev_bank_dataset.csv")
print(bank_df.Bank.nunique())
np.random.seed(8)


# create partition map with two receiving banks
partition_to_banks = defaultdict(set)
partition_to_banks["bank01"] = {"BANKUSUS"}
partition_to_banks["bank02"] = {"BANKDEDE"}
partitioned_banks = {"BANKUSUS", "BANKDEDE"}

# randomly assign sending banks to remaining partitions
N_PARTITIONS = 2
banks = bank_df[~bank_df.Bank.isin(partitioned_banks)].Bank.unique()
print("hi", "BANKUSUS" in banks, "BANKDEDE" in banks)

# partition names are 1 indexed. Leave first two partitions for receiving banks.
partition_ids = [(i % N_PARTITIONS) + 3 for i in range(len(banks))]
np.random.shuffle(partition_ids)
bank_to_partition = dict(zip(banks, partition_ids))

# invert dictionary
for bank, partition in bank_to_partition.items():
    partition_to_banks[f"bank{partition:02}"].add(bank)

# inspect
for partition in sorted(partition_to_banks):
    print(f"{partition}: {len(partition_to_banks[partition])} bank(s)")

for partition_name, bank_set in partition_to_banks.items():
    subset = bank_df[bank_df["Bank"].isin(bank_set)]
    for out_dir in [TRAIN, TEST]:
        out_file = out_dir / partition_name / "bank_dataset.csv"
        out_file.parent.mkdir(exist_ok=True, parents=True)
        print(f"Writing {out_file.relative_to(out_file.parents[3])}...")
        subset.to_csv(out_file)