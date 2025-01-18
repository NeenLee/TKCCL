import numpy as np
import pandas as pd

remap_user_id = {}
remap_user = pd.read_csv(r'../Data/Cell_Phones_and_Accessories/2014/remap_reviewer.txt', sep=' ', header='infer')
array_remap_user = remap_user.values[0::, 0::]
for i in range(0, len(array_remap_user)):
    remap_user_id[array_remap_user[i][1]] = array_remap_user[i][0]

detected_data = pd.read_csv(r'user_item_anomalyScore.csv', sep=',', header='infer')
array_detected = detected_data.values
sorted_array_detect = np.argsort(array_detected[:, 1])[::-1]
sorted_array_detect = array_detected[sorted_array_detect]
