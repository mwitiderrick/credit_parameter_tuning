from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A6"]]

   # rename feature
   df.rename(columns={" A6": "feature_five"}, inplace=True) # dataset stored A5 as A6

   # change the missing value to None
   df["feature_five"].replace({"?":"None"}, inplace=True)

   # convert strings to categorical values
   df["feature_five"] = df["feature_five"].astype("category")
   df["feature_five"] = df["feature_five"].cat.codes

   df = df[["ID", "feature_five"]]

   return df