import gc
import math
import sqlite3
import numpy as np
import pandas as pd
import tqdm
from TSGD.data_factory.loadRemap2ViewData import KGLoader, StructureLoader
from TSGD.pretrain.pretrain_embedding import pretrain_embedding
import os
from TSGD.output.read_result import read_result, read_result_Yelp


def sort_tuples_by_second_element(tuples_array):
    # 使用sorted函数和lambda表达式来排序
    return sorted(tuples_array, key=lambda x: x[1])


def split(x, split_indices):
    # 分割操作
    parts = []
    start = 0
    for index in split_indices:
        if index > 0:
            parts.append(x[start:index])
        start = index
    parts.append(x[start:])
    # if start < len(x):
    #     parts.append(x[start + 1:])  # 添加最后一个部分
    return parts


def split2(x, split_indices):
    x = np.array(x)
    new_x = np.split(x, split_indices)
    print(new_x)
    return new_x


def magnitude(data_points):
    # 计算向量的差
    mod_list = []
    x_values = [float(point[1]) for point in data_points]
    y_values = [float(point[0]) for point in data_points]
    time_diffs = (np.diff(x_values)) / 86400
    score_diffs = np.diff(y_values)
    for i, time_diff in enumerate(time_diffs):
        # 计算向量的模
        mod = math.sqrt(time_diff ** 2 + score_diffs[i] ** 2)
        # for time_diff, score_diff in zip(time_diffs, score_diffs):
        #     # 计算向量的模
        #     mod = math.sqrt(time_diff ** 2 + score_diff ** 2)
        mod_list.append(mod)
    return mod_list


def find_Max_Min_mod(cursor):
    product_set = set()
    max_mod_list = []
    min_mod_list = []
    # min_mod = max_mod = num_iter = 0
    # Yelp_New York
    # cursor.execute("select prod_id from metadata where date BETWEEN '2014-01-01' AND '2014-12-31'")

    # Yelp_Zip
    # cursor.execute("select prod_id from metadata where date BETWEEN '2014-01-01' AND '2014-12-31'")

    # Yelp_Chi
    cursor.execute("select productID_map from meta_yelpResData")

    asins = cursor.fetchall()
    for asin in asins:
        product_set.add(asin[0])
    for pro in tqdm.tqdm(product_set):
        # Yelp_New York
        # cursor.execute(
        #     "select rating, strftime('%s', date) from metadata where prod_id=? and date BETWEEN '2014-01-01' AND '2014-12-31'",
        #     (pro,))
        # Yelp_Zip
        # cursor.execute(
        #     "select rating, strftime('%s', date) from metadata where prod_id=? and date BETWEEN '2014-01-01' AND '2014-12-31'",
        #     (pro,))
        # Yelp_Chi
        cursor.execute(
            "select ratings, strftime('%s',formatted_date) from meta_yelpResData where productID_map=?",
            (pro,))
        row = cursor.fetchall()
        sorted_tuples = sort_tuples_by_second_element(row)
        if len(sorted_tuples) > 2:
            # 这里可以添加进一步处理sorted_tuples的逻辑
            mod_list = magnitude(sorted_tuples)
            log_mod_list_values = [math.log1p(x) if x != 0 else 0 for x in mod_list]
            max_mod_list.append(max(log_mod_list_values))
            min_mod_list.append(min(log_mod_list_values))
        # if num_iter == len(product_set):
    max_mod = max(max_mod_list)
    min_mod = min(min_mod_list)
    return max_mod, min_mod


class struct_kg_views:
    def __init__(self, database_path, file_path):
        self.database_path = database_path
        self.normal_mod_list_all = set()
        # 使用连接池来避免每次查询都重新连接数据库
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.file_path = file_path
        self.fw = open(self.file_path + 'Burst_Dataset.txt', 'w')
        self.fw.write('Product_id' + ' ' + 'Rating' + ' ' + 'Timestamp' + ' ' + 'Reviewer_id' + '\n')
        self.fw_KG = open(self.file_path + 'KG.txt', 'w')
        self.fw_KG.write('Product_id' + ' ' + 'Rating' + ' ' + 'Timestamp' + '\n')
        self.fw_Struct = open(self.file_path + 'Struct.txt', 'w')
        self.fw_Struct.write('Product_id' + 'Reviewer_id' + '\n')
        self.fw_Struct_KG_view = open(self.file_path + 'Struct_KG.txt', 'w')
        self.fw_Struct_KG_view.write(
            '{product:[reviewer1, reviewer2, Rating1, Rating2, timeStamp1, timestamp2, vector mod for rating and '
            'timestamp of difference]}' + '\n')
        self.max_mod, self.min_mod = find_Max_Min_mod(self.cursor)
        # self.max_mod, self.min_mod = 7.92, 0

    def list_split(self, pro, sorted_tuples, split_indices):
        # sorted_tuples is the Overall, UnixReviewTime, reviewerID sort by timeStamp
        # pro is the product of the sort_tuples
        # split_indices is split indexs list for sort_tuples
        sessions = split(sorted_tuples, split_indices)
        # print("分割后session：", sessions)
        # print("分割后session：len", len(sessions))
        for j, session in enumerate(sessions):
            # print(f"第{j}个分割后：", session)
            if len(session) > 1:
                for w in range(0, len(session) - 1):
                    self.fw.write(str(pro) + ' ' + str(session[w][0]) + ' ' + str(session[w][1]) + ' ' + str(
                        session[w][2]) + '\n')
                    self.fw.flush()
                    self.fw_KG.write(str(pro) + ' ' + str(session[w][0]) + ' ' + str(session[w][1]) + '\n')
                    self.fw_KG.flush()
                    self.fw_Struct.write(str(pro) + ' ' + str(session[w][1]) + '\n')
                    self.fw_Struct.flush()

    def Struct_KG_view(self, time_session_threshold):
        print(self.max_mod, self.min_mod)
        product_set = set()
        # Yelp_New York
        # self.cursor.execute("select prod_id from metadata where date BETWEEN '2014-01-01' AND '2014-12-31'")
        # YelpZip
        # self.cursor.execute("select prod_id from metadata where date BETWEEN '2014-01-01' AND '2014-12-31'")
        # YelpChi
        self.cursor.execute("select productID_map from meta_yelpResData")
        asins = self.cursor.fetchall()
        for asin in asins:
            product_set.add(asin[0])
        # print(product_set)
        print("normal its...")
        # for pro in tqdm.tqdm(product_set):
        for pro in product_set:
            product_reviewer_magnitude = {}
            normal_mod_list = []
            split_indices = []
            # Yelp_New York
            # self.cursor.execute(
            #     "select rating, strftime('%s', date), user_id from metadata where prod_id=? and date BETWEEN '2014-01-01' AND '2014-12-31'",
            #     (pro,))
            # YelpZip
            # self.cursor.execute(
            #     "select  rating, strftime('%s', date), user_id from metadata where prod_id=? and date BETWEEN '2014-01-01' AND '2014-12-31'",
            #     (pro,))
            # YelpChi
            self.cursor.execute(
                "select ratings, strftime('%s',formatted_date), reviewerID_map from meta_yelpResData where productID_map=?",
                (pro,))
            # and formatted_date BETWEEN '2012-01-01' AND '2012-12-31'
            row = self.cursor.fetchall()
            sorted_tuples = sort_tuples_by_second_element(row)
            if len(sorted_tuples) > 1:
                # print(sorted_tuples)
                # 这里可以添加进一步处理sorted_tuples的逻辑
                mod_list = magnitude(sorted_tuples)
                log_mod_list_values = [math.log1p(x) if x != 0 else 0 for x in mod_list]
                for i, log_mod in enumerate(log_mod_list_values):
                    log_mod = (log_mod - self.min_mod) / (self.max_mod - self.min_mod)
                    normal_mod_list.append(log_mod)
                    self.normal_mod_list_all.add(log_mod)
                    if log_mod > time_session_threshold:
                        split_indices.append(i + 1)
                # print("归一化列表: ", normal_mod_list)
                # print("len normal_mod_list", len(normal_mod_list))
                # print("split_indices", split_indices)
                # print("len split_indices", len(split_indices))
                # print("分割前session：", sorted_tuples)
                # print("分割前session：len ", len(sorted_tuples))
                product_reviewer_magnitude[pro] = []
                i = 0
                while i < len(sorted_tuples) - 1:
                    product_reviewer_magnitude[pro].append(
                        (sorted_tuples[i][2], sorted_tuples[i + 1][2], sorted_tuples[i][0]
                         , sorted_tuples[i + 1][0], sorted_tuples[i][1], sorted_tuples[i + 1][1]))
                    # product_reviewer_magnitude format as -{'product':[reviewer1,reviewer2,Rating1,Rating2,timeStamp1,
                    # timestamp2,vector mod for rating and timestamp of difference]}
                    i += 1
                for i, reviewer_tuple in enumerate(product_reviewer_magnitude[pro]):
                    normal_mod_list[i] = "{:.4f}".format(normal_mod_list[i])
                    product_reviewer_magnitude[pro][i] = (*reviewer_tuple, normal_mod_list[i])
                # print(product_reviewer_magnitude)
                self.fw_Struct_KG_view.write(str(product_reviewer_magnitude) + '\n')
                self.list_split(pro, sorted_tuples, split_indices)
        return self.normal_mod_list_all
        # sessions.clear()
        # self.split_array(normal_integral_area)

        # print(normal_differences)
        # print('评分差序列', difference)

    def close(self):
        self.fw.close()
        self.fw_KG.close()
        self.fw_Struct.close()
        self.fw_Struct_KG_view.close()
        # 确保在结束时关闭连接
        self.conn.close()

    def remap(self, fr_path, KG_path, Structure_path, reviewer_path, product_path):
        fr = open(fr_path, 'r')
        remap_KG = open(KG_path, 'w')
        remap_Structure = open(Structure_path, 'w')
        remap_product = open(product_path, 'w')
        remap_reviewer = open(reviewer_path, 'w')

        remap_KG.write('product_id rating timestamp\n')
        remap_Structure.write('remap_product_id remap_reviewer_id\n')
        remap_product.write('product_id remap_id \n')
        remap_reviewer.write('reviewer_id remap_id \n')
        product_ids = []
        user_ids = []
        timestamps = []
        ratings = []
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
            remap_reviewer.write(str(remap[0]) + ' ' + str(remap[1]) + '\n')

        for remap in zip(set(product_ids), set(numeric_product_ids)):
            remap_product.write(str(remap[0]) + ' ' + str(remap[1]) + '\n')

        for remap in zip(numeric_product_ids, ratings, timestamps):
            remap_KG.write(str(remap[0]) + ' ' + str(remap[1]) + ' ' + str(remap[2]) + '\n')

        for remap in zip(numeric_product_ids, numeric_user_ids):
            remap_Structure.write(str(remap[1]) + ' ' + str(remap[0]) + '\n')
        fr.close()
        remap_product.close()
        remap_reviewer.close()
        remap_KG.close()
        remap_Structure.close()


if __name__ == '__main__':
    data_path = '../Data/Yelp_Chi/'  # 换数据修改 别忘了修改main函数中的预训练
    database_path = 'G:\\DB\\YelpChi.db'  # 换数据修改
    result_path = '../output/Yelp_Chi/'  # 换数据修改

    frp = data_path + 'Burst_Dataset.txt'
    KGp = data_path + 'remap_KG.txt'
    Sp = data_path + 'remap_Structure.txt'  # structure file train
    rp = data_path + 'remap_reviewer.txt'
    prop = data_path + 'remap_product.txt'
    kg_final_path = data_path + 'kg_final.txt'  # kg_final.txt (user item days) to write for train
    sp_final_path = data_path + 'train.txt'  # train.txt (user item) to write for train
    ss_final_path = data_path + 'test.txt'  # train.txt (user item) to write for train

    pretrain_embedding_path = '../pretrain/YelpChi/cp.npz'
    detected_path = result_path + 'user_item_anomalyScore_0.7_1.csv'  # detected_spammer_path
    parm_path = result_path + 'parm_precision_temperature_0.7_1.csv'

    test = struct_kg_views(database_path, data_path)
    print("differences the Product sequences...")
    normal_mod_list_all = test.Struct_KG_view(0)
    normal_mod_list_all = sorted(normal_mod_list_all)
    print(normal_mod_list_all)
    # 定义接近程度的阈值
    proximity_threshold = 0.02
    # 初始化结果数组
    # result_mod = [normal_mod_list_all[0]]
    result_mod = [1, 1]
    # 遍历数组并合并阈值
    # for i in range(1, len(normal_mod_list_all)):
    #     if normal_mod_list_all[i] - result_mod[-1] > proximity_threshold:
    #         result_mod.append(normal_mod_list_all[i])

    parm_precision_result = {'mod': result_mod, 'precision': [], 'recall': [], 'f1': []}
    num_parm = 1
    print(result_mod)
    for mod in result_mod:
        print(f'Processing num mod: {num_parm}, length: {len(result_mod)}')
        print('************************************************************')
        num_parm += 1
        test.Struct_KG_view(mod)
        test.remap(frp, KGp, Sp, rp, prop)
        KGLoader(KGp, kg_final_path)
        StructureLoader(Sp, sp_final_path, ss_final_path)
        pretrain_embedding(Sp, pretrain_embedding_path)
        # test.close()  # 使用完毕后关闭数据库连接
        os.system(r'python main.py')
        precision, recall, f1, result = read_result_Yelp(database_path, rp, detected_path)
        parm_precision_result['precision'].append(precision)
        parm_precision_result['recall'].append(recall)
        parm_precision_result['f1'].append(f1)
        # break
    max_precision = max(parm_precision_result['precision'])
    max_precision_index = parm_precision_result['precision'].index(max_precision)
    best_mod = parm_precision_result['mod'][max_precision_index]
    print(f'best mod{best_mod},get best precision{max_precision}')
    df_parm = pd.DataFrame(parm_precision_result)
    df_parm.to_csv(parm_path, index=True, sep=',')
