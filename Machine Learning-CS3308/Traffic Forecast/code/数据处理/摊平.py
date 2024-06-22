import pandas as pd

df = pd.read_csv('data/temp/temp_test.csv')

df['index'] = (df['iu_ac'] - 5 ) * 8760 * 3

df = df.drop(columns=['iu_ac'])

df.to_csv('data/temp/for_test.csv', index=False)

df = pd.read_csv('data/temp/time_train.csv')

df['index'] = (df['iu_ac'] - 5 ) * 8760 * 3

df = df.drop(columns=['iu_ac'])

df.to_csv('data/temp/for_train.csv', index=False)


