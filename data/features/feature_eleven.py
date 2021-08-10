from typing import Any
from layer import Dataset
from sklearn.preprocessing import StandardScaler

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A11"]]

   # rename feature
   df.rename(columns={" A11": "feature_eleven"}, inplace=True)

   # normalize the values
   scaler = StandardScaler()
   df["feature_eleven"] = scaler.fit_transform(df[["feature_eleven"]])
   df = df[["ID", "feature_eleven"]]

   return df