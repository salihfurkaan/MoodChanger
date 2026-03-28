import pandas as pd

df = pd.read_csv('analytics_pipeline_output.csv')
print('pipeline', df.shape)
print('cols', list(df.columns))
raw = pd.read_csv('raw_wearable_data.csv')
print('raw', raw.shape)
