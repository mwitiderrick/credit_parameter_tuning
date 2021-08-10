from typing import Any
from layer import Dataset
import pandas as pd

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A9"]]

   # rename feature
   df.rename(columns={" A9": "feature_nine"}, inplace=True)

   # one hot encoding
   features = pd.get_dummies(df["feature_nine"], drop_first=True).rename(columns={"t": "feature_nine"})
   df = pd.concat([df["ID"], features], axis=1)
   df = df[["ID", "feature_nine"]]
   return df