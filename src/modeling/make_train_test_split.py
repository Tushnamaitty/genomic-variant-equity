# src/modeling/make_train_test_split.py

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = Path("data/processed/ml/chr22_ml_dataset.csv")
OUTDIR = Path("data/processed/ml/splits")

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["is_resolved"],
        random_state=RANDOM_STATE,
    )

    OUTDIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTDIR / "train.csv"
    test_path = OUTDIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain target:")
    print(train_df["is_resolved"].value_counts())
    print(train_df["is_resolved"].value_counts(normalize=True))

    print("\nTest target:")
    print(test_df["is_resolved"].value_counts())
    print(test_df["is_resolved"].value_counts(normalize=True))

    print("\nSaved:")
    print(train_path)
    print(test_path)


if __name__ == "__main__":
    main()