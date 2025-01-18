import pandas as pd
import sqlite3

con = sqlite3.connect('G:\\DB\\YelpZip.db')
cursor = con.cursor()
fw = open('YelpZip_1000.txt', 'w')
df = pd.read_csv('user_item_anomalyScore_0.7.csv')
sorted_df = df.sort_values(by='score', ascending=False)
print(sorted_df['user_item_id'])
for i in range(0, 1000):
    reviewer = str(sorted_df['user_item_id'][i])
    cursor.execute('select label from metadata where user_id=\"' + reviewer + '\"')
    rows = cursor.fetchall()
    if (-1,) in rows:
        fw.write('1' + '\n')
    else:
        fw.write('0' + '\n')
