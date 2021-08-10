from typing import Any
from layer import Dataset

def build_feature(sdf: Dataset("credit_approval_data")) -> Any:
   df = sdf.to_pandas()
   df = df[["ID", " A16"]]

   # renaming column name
   df.rename(columns={" A16": "approved"}, inplace=True)

   # changing the values
   target_map = {"+": 1, "-": 0}
   df["approved"] = df["approved"].map(target_map)

   df = df[["ID", "approved"]]
   return df

