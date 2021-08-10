from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A13"]]

   # rename feature
   df.rename(columns={" A13": "feature_thirteen"}, inplace=True)

   # convert strings to categorical values
   df["feature_thirteen"] = df["feature_thirteen"].astype("category")
   df["feature_thirteen"] = df["feature_thirteen"].cat.codes

   df = df[["ID", "feature_thirteen"]]

   return df