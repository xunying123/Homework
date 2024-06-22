import pandas as pd
import os

df = pd.read_csv('data/loop_sensor_train.csv')

df['t_1h'] = pd.to_datetime(df['t_1h'])

base_time = pd.Timestamp('2022-01-01 00:00:00')

df['index'] = (df['t_1h'] - base_time) / pd.Timedelta(hours=1)

df = df.drop(columns=['t_1h'])

df = df.drop(columns=['etat_barre'])

os.makedirs('data/temp', exist_ok=True)

df.to_csv('data/temp/time_train.csv', index=False)

df = pd.read_csv('data/loop_sensor_test_x.csv')

df['t_1h'] = pd.to_datetime(df['t_1h'])

base_time = pd.Timestamp('2022-01-01 00:00:00')

df['index'] = (df['t_1h'] - base_time) / pd.Timedelta(hours=1)

df = df.drop(columns=['t_1h'])

df = df.drop(columns=['etat_barre'])

os.makedirs('data/temp', exist_ok=True)

df.to_csv('data/temp/time_test.csv', index=False)

