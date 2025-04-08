# convert rt_data/PXD006109/PXD006109_Cerebellum_rt_add_mox_all_rt_range_3_train.tsv to PXD006109/PXD006109_train.csv
# x and y column change to sequence and RT column separated by comma

import pandas as pd
import numpy as np
import os

def convert_data_format(input_file, output_file):
    data = pd.read_csv(input_file, sep='\t')
    data['sequence'] = data['x']
    data['RT'] = data['y']
    data = data.drop(['x', 'y'], axis=1)
    data.to_csv(output_file, index=False)

if __name__ == '__main__':
    input_file = 'rt_data/SAL00141/SAL00141_test.tsv'
    output_file = 'SAL00141/SAL00141_test.csv'
    convert_data_format(input_file, output_file)