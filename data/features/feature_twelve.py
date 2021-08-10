from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A12"]]

   # rename feature
   df.rename(columns={" A12": "feature_twelve"}, inplace=True)

   # one hot encoding
   features = pd.get_dummies(df["feature_twelve"], drop_first=True).rename(columns={"t": "feature_twelve"})
   df = pd.concat([df["ID"], features], axis=1)
   df = df[["ID", "feature_twelve"]]
   return df