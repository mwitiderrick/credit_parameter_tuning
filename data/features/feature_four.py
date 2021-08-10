from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A4"]]

   # rename feature
   df.rename(columns={" A4": "feature_four"}, inplace=True)

   # change the missing value to None
   df["feature_four"].replace({"?":"None"}, inplace=True)

   # convert strings to categorical values
   df["feature_four"] = df["feature_four"].astype("category")
   df["feature_four"] = df["feature_four"].cat.codes

   df = df[["ID", "feature_four"]]

   return df