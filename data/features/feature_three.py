from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
    df = sdf.to_pandas()
    df = df[["ID", " A3"]]

    # rename feature
    df.rename(columns={" A3": "feature_three"}, inplace=True)
    df = df[["ID", "feature_three"]]
    return df