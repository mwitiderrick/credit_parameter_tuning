from typing import Any
from layer import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A2"]]

   # rename feature
   df.rename(columns={" A2": "feature_two"}, inplace=True)

   # change the missing value to None
   df["feature_two"].replace({"?":None}, inplace=True)

   # convert to a numeric feature
   df["feature_two"] = df["feature_two"].apply(pd.to_numeric)

   # fill missing value with mean
   df["feature_two"].fillna(value=df["feature_two"].mean(), inplace=True)

   # normalize the values
   scaler = StandardScaler()
   df["feature_two"] = scaler.fit_transform(df[["feature_two"]])
   df = df[["ID", "feature_two"]]

   return df