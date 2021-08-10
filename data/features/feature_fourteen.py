from typing import Any
from layer import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A14"]]

   # rename feature
   df.rename(columns={" A14": "feature_fourteen"}, inplace=True)

   # change the missing value to None
   df["feature_fourteen"].replace({"?":None}, inplace=True)

   # convert to a numeric feature
   df["feature_fourteen"] = df["feature_fourteen"].apply(pd.to_numeric)

   # fill missing value with mean
   df["feature_fourteen"].fillna(value=df["feature_fourteen"].mean(), inplace=True)

   # normalize the values
   scaler = StandardScaler()
   df["feature_fourteen"] = scaler.fit_transform(df[["feature_fourteen"]])
   df = df[["ID", "feature_fourteen"]]

   return df