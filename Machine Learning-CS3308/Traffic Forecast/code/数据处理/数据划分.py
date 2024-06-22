import csv
import os

input_file = 'data/temp/time_test.csv'
output_dir = 'data/temp/source'

os.makedirs(output_dir, exist_ok=True)

file_pointers = {}

try:
    with open(input_file, 'r') as infile:
        reader = csv.DictReader(infile)
        
        for row in reader:
            id_value = row['iu_ac']  
            
            if id_value not in file_pointers:
                output_file = os.path.join(output_dir, f'{id_value}.csv')
                file_pointers[id_value] = open(output_file, 'w', newline='')
                writer = csv.DictWriter(file_pointers[id_value], fieldnames=reader.fieldnames)
                writer.writeheader()  
            
            writer = csv.DictWriter(file_pointers[id_value], fieldnames=reader.fieldnames)
            writer.writerow(row)
finally:
    for fp in file_pointers.values():
        fp.close()
