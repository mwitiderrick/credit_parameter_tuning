from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A10"]]

   # rename feature
   df.rename(columns={" A10": "feature_ten"}, inplace=True)

   # change the missing value to None
   df["feature_ten"].replace({"?":"None"}, inplace=True)

   # convert strings to categorical values
   df["feature_ten"] = df["feature_ten"].astype("category")
   df["feature_ten"] = df["feature_ten"].cat.codes

   df = df[["ID", "feature_ten"]]

   return df