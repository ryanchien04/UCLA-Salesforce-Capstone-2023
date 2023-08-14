import pandas as pd
import pyarrow.parquet as pq

table = pq.read_table('../snip.parquet', columns=['content'])
print(table.to_pandas())