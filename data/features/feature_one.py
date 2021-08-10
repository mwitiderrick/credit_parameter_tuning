from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", "A1"]]

   # rename feature
   df.rename(columns={"A1": "feature_one"}, inplace=True)

   # change the missing value to None
   df["feature_one"].replace({"?":None}, inplace=True)

   # one hot encoding
   features = pd.get_dummies(df["feature_one"], drop_first=True).rename(columns={"b": "feature_one"})
   df = pd.concat([df["ID"], features], axis=1)
   df = df[["ID", "feature_one"]]
   return df