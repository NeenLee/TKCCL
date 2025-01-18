import math
import sqlite3
import numpy as np
import tqdm


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
    x_values = [point[1] for point in data_points]
    y_values = [point[0] for point in data_points]
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
    cursor.execute('select asin from Cell_Phones_and_Accessories')
    asins = cursor.fetchall()
    for asin in asins:
        product_set.add(asin[0])
    for pro in tqdm.tqdm(product_set):
        cursor.execute("select Overall, UnixReviewTime from Cell_Phones_and_Accessories where asin=?",
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


def remap(file_path):
    product_ids = []
    user_ids = []
    timestamps = []
    ratings = []

    fr = open(file_path + 'Burst_Dataset.txt', 'r')
    fw_remap_product = open(file_path + 'remap_product.txt', 'w')
    fw_remap_product.write('product_id remap_id \n')

    fw_remap_reviewer = open(file_path + 'remap_reviewer.txt', 'w')
    fw_remap_reviewer.write('reviewer_id remap_id \n')

    fw_remap_KG = open(file_path + 'remap_KG.txt', 'w')
    fw_remap_KG.write('product_id rating timestamp\n')

    fw_remap_Structure = open(file_path + 'remap_Structure.txt', 'w')
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
        fw_remap_Structure.write(str(remap[0]) + ' ' + str(remap[1]) + '\n')
    fr.close()
    fw_remap_product.close()
    fw_remap_reviewer.close()
    fw_remap_KG.close()
    fw_remap_Structure.close()


class struct_kg_views:
    def __init__(self, database_path, time_session_threshold):
        self.database_path = database_path
        self.normal_mod_list_all = set()
        self.time_session_threshold = time_session_threshold
        # 使用连接池来避免每次查询都重新连接数据库
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.file_path = 'Data/Cell_Phones_and_Accessories/All\\'
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

    def list_split(self, pro, sorted_tuples, split_indices):
        # sorted_tuples is the Overall, UnixReviewTime, reviewerID sort by timeStamp
        # pro is the product of the sort_tuples
        # split_indices is split indexs list for sort_tuples
        sessions = split(sorted_tuples, split_indices)
        # print("分割后session：", sessions)
        # print("分割后session：len", len(sessions))
        for j, session in enumerate(sessions):
            # print(f"第{j}个分割后：", session)
            if len(session) > 2:
                for w in range(0, len(session) - 1):
                    self.fw.write(str(pro) + ' ' + str(session[w][0]) + ' ' + str(session[w][1]) + ' ' + str(
                        session[w][2]) + '\n')
                    self.fw.flush()
                    self.fw_KG.write(str(pro) + ' ' + str(session[w][0]) + ' ' + str(session[w][1]) + '\n')
                    self.fw_KG.flush()
                    self.fw_Struct.write(str(pro) + ' ' + str(session[w][1]) + '\n')
                    self.fw_Struct.flush()

    def Struct_KG_view(self):
        max_mod = 5.86
        min_mod = 0
        product_set = set()
        self.cursor.execute('select asin from Cell_Phones_and_Accessories_2010')
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
            self.cursor.execute(
                "select  Overall, UnixReviewTime, reviewerID from Cell_Phones_and_Accessories_2010 where asin=?",
                (pro,))
            row = self.cursor.fetchall()
            sorted_tuples = sort_tuples_by_second_element(row)
            if len(sorted_tuples) > 2:
                # print(sorted_tuples)
                # 这里可以添加进一步处理sorted_tuples的逻辑
                mod_list = magnitude(sorted_tuples)
                log_mod_list_values = [math.log1p(x) if x != 0 else 0 for x in mod_list]
                for i, mod in enumerate(log_mod_list_values):
                    mod = (mod - min_mod) / (max_mod - min_mod)
                    normal_mod_list.append(mod)
                    self.normal_mod_list_all.add(mod)
                    if mod > self.time_session_threshold:
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


if __name__ == '__main__':
    test = struct_kg_views('G:\\DB\\amazonReviews.db', 0.0)
    print("differences the asin sequences...")
    normal_mod_list_all = test.Struct_KG_view()
    print(len(normal_mod_list_all))
    test.close()
    remap(test.file_path)
    # test.close()  # 使用完毕后关闭数据库连接
