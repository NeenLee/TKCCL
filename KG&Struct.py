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


class struct_kg_views:
    def __init__(self, database_path, time_session_threshold):
        self.database_path = database_path
        self.time_session_threshold = time_session_threshold
        # 使用连接池来避免每次查询都重新连接数据库
        self.conn = sqlite3.connect(self.database_path, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self.fw = open('Data/Cell_Phones_and_Accessories/2010\\Burst_Dataset.txt', 'w')
        self.fw.write('Product_id' + ' ' + 'Rating' + ' ' + 'Timestamp' + ' ' + 'Reviewer_id' + '\n')
        self.fw_KG = open('Data/Cell_Phones_and_Accessories/2010\\KG.txt', 'w')
        self.fw_KG.write('Product_id' + ' ' + 'Rating' + ' ' + 'Timestamp' + '\n')
        self.fw_Struct = open('Data/Cell_Phones_and_Accessories/2010\\Struct.txt', 'w')
        self.fw_Struct.write('Product_id' + 'Reviewer_id' + '\n')

    def KG_view(self):
        product_set = set()
        max_mod_list = []
        min_mod_list = []
        # min_mod = max_mod = num_iter = 0
        self.cursor.execute('select asin from 2010')
        asins = self.cursor.fetchall()
        for asin in asins:
            product_set.add(asin[0])
        for pro in tqdm.tqdm(product_set):
            self.cursor.execute("select Overall, UnixReviewTime from 2010 where asin=?",
                                (pro,))
            row = self.cursor.fetchall()
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

        print("normal its...")
        for pro in tqdm.tqdm(product_set):
            normal_mod_list = []
            split_indices = []
            self.cursor.execute(
                "select  Overall, UnixReviewTime, reviewerID from 2010 where asin=?",
                (pro,))
            row = self.cursor.fetchall()
            sorted_tuples = sort_tuples_by_second_element(row)
            if len(sorted_tuples) > 2:
                # 这里可以添加进一步处理sorted_tuples的逻辑
                mod_list = magnitude(sorted_tuples)
                log_mod_list_values = [math.log1p(x) if x != 0 else 0 for x in mod_list]
                for i, mod in enumerate(log_mod_list_values):
                    mod = (mod - min_mod) / (max_mod - min_mod)
                    normal_mod_list.append(mod)
                    if mod > self.time_session_threshold:
                        split_indices.append(i + 1)
                print("归一化列表: ", normal_mod_list)
                print("len normal_mod_list", len(normal_mod_list))
                print("split_indices", split_indices)
                print("len split_indices", len(split_indices))
                print("分割前session：", sorted_tuples)
                print("分割前session：len ", len(sorted_tuples))
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
        self.fw.close()
        # sessions.clear()
        # self.split_array(normal_integral_area)

        # print(normal_differences)
        # print('评分差序列', difference)

    def close(self):
        # 确保在结束时关闭连接
        self.conn.close()


if __name__ == '__main__':
    test = struct_kg_views('G:\\DB\\amazonReviews.db', 0.6)
    print("differences the asin sequences...")
    test.KG_view()
    test.close()  # 使用完毕后关闭数据库连接
