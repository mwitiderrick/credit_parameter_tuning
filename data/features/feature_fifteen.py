from typing import Any
from layer import Dataset
from sklearn.preprocessing import StandardScaler

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A15"]]

   # rename feature
   df.rename(columns={" A15": "feature_fifteen"}, inplace=True)

   # normalize the values
   scaler = StandardScaler()
   df["feature_fifteen"] = scaler.fit_transform(df[["feature_fifteen"]])
   df = df[["ID", "feature_fifteen"]]

   return df