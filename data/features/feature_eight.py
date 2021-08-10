from typing import Any
from layer import Dataset
from sklearn.preprocessing import StandardScaler

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A8"]]

   # rename feature
   df.rename(columns={" A8": "feature_eight"}, inplace=True)

   # normalize the values
   scaler = StandardScaler()
   df["feature_eight"] = scaler.fit_transform(df[["feature_eight"]])
   df = df[["ID", "feature_eight"]]

   return df