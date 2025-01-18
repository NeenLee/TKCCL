product_ids = []
user_ids = []
timestamps = []
ratings = []
fr = open('../Data/Cell_Phones_and_Accessories/2010/Burst_Dataset.txt', 'r')
fw_remap_product = open('../Data/Cell_Phones_and_Accessories/2010/remap_product.txt', 'w')
fw_remap_product.write('product_id remap_id \n')

fw_remap_reviewer = open('../Data/Cell_Phones_and_Accessories/2010/remap_reviewer.txt', 'w')
fw_remap_reviewer.write('reviewer_id remap_id \n')

fw_remap_KG = open('../Data/Cell_Phones_and_Accessories/2010/remap_KG.txt', 'w')
fw_remap_KG.write('product_id rating timestamp\n')

fw_remap_Structure = open('../Data/Cell_Phones_and_Accessories/2010/remap_Structure.txt', 'w')
fw_remap_Structure.write('remap_product_id remap_reviewer_id\n')

lines = fr.readlines()
for i in range(1, len(lines) - 1):
    line = lines[i].strip('\n').split(' ')
    product_ids.append(line[0])
    ratings.append(line[1])
    timestamps.append(line[2])
    user_ids.append(line[3])
# 创建一个从ID到数字的映射字典
product_id_to_numeric = {id: i for i, id in enumerate(sorted(set(product_ids)))}
user_id_to_numeric = {id: i for i, id in enumerate(sorted(set(user_ids)))}
# 将原始ID列表转换为数字列表
numeric_product_ids = [product_id_to_numeric[id] for id in product_ids]
numeric_user_ids = [user_id_to_numeric[id] for id in user_ids]
# print(numeric_product_ids)  # 输出可能是 [0, 1, 2, 3]
for remap in zip(set(user_ids), set(numeric_user_ids)):
    fw_remap_reviewer.write(str(remap[0]) + ' ' + str(remap[1]) + '\n')

for remap in zip(set(product_ids), set(numeric_product_ids)):
    fw_remap_product.write(str(remap[0]) + ' ' + str(remap[1]) + '\n')

for remap in zip(numeric_product_ids, ratings, timestamps):
    fw_remap_KG.write(str(remap[0]) + ' ' + str(remap[1]) + ' ' + str(remap[2]) + '\n')

for remap in zip(numeric_product_ids, numeric_user_ids):
    fw_remap_Structure.write(str(remap[1]) + ' ' + str(remap[0]) + '\n')
fr.close()
fw_remap_product.close()
fw_remap_reviewer.close()
fw_remap_KG.close()
fw_remap_Structure.close()
