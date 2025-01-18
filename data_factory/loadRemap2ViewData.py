import sys


def KGLoader(KG, kg_final_path):
    fw_st = open(kg_final_path, 'w', encoding='utf-8')
    product_sequence = dict()
    timestamp_list = []
    with open(KG, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            key, value1, value2 = line.split()
            timestamp_list.append(value2)
        min_timestamp = min(timestamp_list)
        for line in lines:
            key, value1, value2 = line.split()
            # print(key, value1, value2)
            if key not in product_sequence:
                product_sequence[key] = []
            rating = value1
            time_diff = (int(value2) - int(min_timestamp)) / 86400
            product_sequence[key].append((rating, str(time_diff)))

        for (key, value) in product_sequence.items():
            for value_tuple in value:
                fw_st.write(key + ' ' + value_tuple[0] + ' ' + value_tuple[1] + '\n')
    fw_st.close()


def StructureLoader(Structure, train_path, test_path):
    fw_st = open(train_path, 'w', encoding='utf-8')
    fw_ss = open(test_path, 'w', encoding='utf-8')
    user_sequence = dict()
    with open(Structure, 'r') as f:
        lines = f.readlines()[1:]
        for line in lines:
            reviewer, product = line.split()
            # 将Reviewer 与 product互换可以形成用户产品或者产品用户序列
            if reviewer not in user_sequence:
                user_sequence[reviewer] = []
            user_sequence[reviewer].append(product)

        for key in user_sequence:
            fw_st.write(key + ' ')
            for value in user_sequence[key]:
                fw_st.write(value + ' ')
            fw_st.write('\n')
        for key in user_sequence:
            fw_ss.write(key + ' ')
            for value in user_sequence[key]:
                fw_ss.write(value + ' ')
            fw_ss.write('\n')
    fw_st.close()
    fw_ss.close()

# if __name__ == '__main__':
#     data_path = '../Data/Cell_Phones_and_Accessories/2010/'
#     kg_file = data_path + 'remap_KG.txt'
#     structure_file = data_path + 'remap_Structure.txt'
#     kfp = data_path + 'kg_final.txt'
#     tp = data_path + 'train.txt'
#     KGLoader(KG=kg_file, kg_final_path=kfp)
#     StructureLoader(Structure=structure_file, train_path=tp)
