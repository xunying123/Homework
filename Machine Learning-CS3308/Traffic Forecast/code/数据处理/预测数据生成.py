import pandas as pd
import sys

file1 = pd.read_csv('data/temp/time_test.csv')

file2 = pd.read_csv('data/temp/time_train.csv')

result = pd.DataFrame()

for index, row in file1.iterrows():
    current_id = row['iu_ac']

    current_time = row['index']
    
    selected_rows = file2[(file2['iu_ac'] == current_id) & (file2['index'] <= current_time)]

    if len(selected_rows) < 24:

        if len(selected_rows) == 0:
            num_missing = 24

            nmnmnm = file2[(file2['iu_ac'] == current_id) & (file2['index'] >= current_time)]

            nmnmnm = nmnmnm.head(24)

            padding_data = pd.DataFrame({col: [nmnmnm.iloc[0]['q']] * num_missing for col in file2.columns if col != 'index'})

            padding_data['index'] = [current_time - i  for i in range(num_missing, 0, -1)]

            padding_data['iu_ac'] = current_id  
        
        else :

            num = 24 - len(selected_rows)

            padding_data = pd.concat([selected_rows.iloc[[0]]] * num, ignore_index=True)

            padding_data['index'] = [selected_rows.iloc[0]['index'] - i for i in range(num, 0, -1)]

        selected_rows = pd.concat([padding_data, selected_rows], ignore_index=True)
    else:

        selected_rows = selected_rows.tail(24)

    result = pd.concat([result, selected_rows])

    print(index)

    sys.stdout.flush()

result.reset_index(drop=True, inplace=True)

result.to_csv('data/temp/temp_test.csv', index=False)




