import pandas as pd

df = pd.read_csv('data/temp/time_test.csv')

unique_values = df['iu_ac'].unique()

unique_df = pd.DataFrame({'id': unique_values})

unique_df.to_csv('data/temp/id.csv', index=False)
