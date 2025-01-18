import pandas as pd
import numpy as np
import sqlite3


def read_result(remap_user_path, detected_path):
    result = {
        'index': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    database_path = 'G:\\DB\\amazonReviews.db'
    conn = sqlite3.connect(database_path, check_same_thread=False)
    cursor = conn.cursor()
    # 将结果写入csv文件：result_user_id ,result_remap_user_id, sorted_score
    # 分别是:原始user id,remap user id，和已经排序的异常分数。
    result_user_id, result_remap_user_id, sorted_score = [], [], []
    result_spammer = {'result_user_id': [], 'result_remap_user_id': [], 'sorted_score': []}

    remap_user_id = {}
    remap_user = pd.read_csv(remap_user_path, sep=' ', header='infer')
    array_remap_user = remap_user.values[0::, 0::]
    for i in range(0, len(array_remap_user)):
        remap_user_id[array_remap_user[i][1]] = array_remap_user[i][0]

    detected_data = pd.read_csv(detected_path, sep=',', header='infer')
    array_detected = detected_data.values
    sorted_array_detect = np.argsort(array_detected[:, 1])[::-1]
    # print(sorted_array_detect)
    sorted_array_detect = array_detected[sorted_array_detect]
    # print(sorted_array_detect)
    t = (1.0,)  # 查询出来的标签
    sum_reviewer = 0
    sum_spam_reviewer = 0
    index = 0
    for i in range(0, len(sorted_array_detect)):
        # print(f'user id = {remap_user_id[sorted_array_detect[i][0]]},'
        #       f'sorted anomaly score = {sorted_array_detect[i][1]}')
        result_spammer['result_user_id'].append(remap_user_id[sorted_array_detect[i][0]])
        result_spammer['result_remap_user_id'].append(int(sorted_array_detect[i][0]))
        result_spammer['sorted_score'].append(sorted_array_detect[i][1])
        index += 1
        x = str(int(sorted_array_detect[i][0]))
        # print(type(sorted_array_detect[i][0]))
        reviewerID = str(remap_user_id[sorted_array_detect[i][0]])
        cursor.execute(
            "select class from Cell_Phones_and_Accessories_2014 where reviewerID=?",
            (reviewerID,))
        rows = cursor.fetchall()
        for row in rows:
            if row[0] == 1:
                sum_spam_reviewer += 1
                break
        sum_reviewer += 1
        result['index'].append(index)
        result['precision'].append(sum_spam_reviewer / sum_reviewer)
        p = sum_spam_reviewer / sum_reviewer
        result['recall'].append(sum_spam_reviewer / 114883)
        r = sum_spam_reviewer / 114883
        if p + r == 0.0:
            result['f1'].append(0)
        else:
            result['f1'].append((2 * p * r) / (p + r))

        if sum_reviewer == 4000:  # top-x reviewer end
            break
    precision = sum_spam_reviewer / sum_reviewer
    recall = sum_spam_reviewer / 114883
    if precision + recall == 0.0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    dataframe = pd.DataFrame(result_spammer)
    dataframe.to_csv('result spammers.csv', index=False, sep=',')
    # 先不将result_spammer写入文件
    print(len(result['precision']))
    print(len(result['index']))
    return precision, recall, f1, result


# data_path = '../Data/Cell_Phones_and_Accessories/2014/'
# result_path = '../output/Amazon_CPA/2014/'
# rp = data_path + 'remap_reviewer.txt'
# detected_path = result_path + 'user_item_anomalyScore_0.7_0.csv'
# read_result(rp, detected_path)


def read_result_Yelp(database_path, remap_user_path, detected_path):
    result = {
        'index': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    conn = sqlite3.connect(database_path, check_same_thread=False)
    cursor = conn.cursor()
    # 将结果写入csv文件：result_user_id ,result_remap_user_id, sorted_score
    # 分别是:原始user id,remap user id，和已经排序的异常分数。
    result_user_id, result_remap_user_id, sorted_score = [], [], []
    result_spammer = {'result_user_id': [], 'result_remap_user_id': [], 'sorted_score': []}

    remap_user_id = {}
    remap_user = pd.read_csv(remap_user_path, sep=' ', header='infer')
    array_remap_user = remap_user.values[0::, 0::]
    for i in range(0, len(array_remap_user)):
        remap_user_id[array_remap_user[i][1]] = array_remap_user[i][0]

    detected_data = pd.read_csv(detected_path, sep=',', header='infer')
    array_detected = detected_data.values
    sorted_array_detect = np.argsort(array_detected[:, 1])[::-1]
    # print(sorted_array_detect)
    sorted_array_detect = array_detected[sorted_array_detect]
    # print(sorted_array_detect)
    t = (1.0,)  # 查询出来的标签
    sum_reviewer = 0
    sum_spam_reviewer = 0
    index = 0
    for i in range(0, len(sorted_array_detect)):
        # print(f'user id = {remap_user_id[sorted_array_detect[i][0]]},'
        #       f'sorted anomaly score = {sorted_array_detect[i][1]}')
        result_spammer['result_user_id'].append(remap_user_id[sorted_array_detect[i][0]])
        result_spammer['result_remap_user_id'].append(int(sorted_array_detect[i][0]))
        result_spammer['sorted_score'].append(sorted_array_detect[i][1])
        index += 1
        # x= str(int(sorted_array_detect[i][0]))
        # print(type(sorted_array_detect[i][0]))
        reviewerID = str(int(remap_user_id[sorted_array_detect[i][0]]))
        # print(index, reviewerID)
        # YelpNYC or YelpZip
        if database_path == 'G:\\DB\\YelpZip.db' or database_path == 'G:\\DB\\YelpNYC.db':
            cursor.execute(
                "select label from metadata where user_id=?",
                (reviewerID,))
        # YelpChi
        else:
            cursor.execute(
                "select Label from meta_yelpResData where reviewerID_map=?",
                (reviewerID,))
        rows = cursor.fetchall()
        for row in rows:
            if row[0] == -1 or row[0] == 'Y':
                sum_spam_reviewer += 1
                break
        sum_reviewer += 1
        result['index'].append(index)
        result['precision'].append(sum_spam_reviewer / sum_reviewer)
        result['index'].append(index)
        result['precision'].append(sum_spam_reviewer / sum_reviewer)
        p = sum_spam_reviewer / sum_reviewer
        result['recall'].append(sum_spam_reviewer / 7678)
        r = sum_spam_reviewer / 7678
        if p + r == 0.0:
            result['f1'].append(0)
        else:
            result['f1'].append((2 * p * r) / (p + r))

        if sum_reviewer == 4000:  # top-x reviewer end
            break
    precision = sum_spam_reviewer / sum_reviewer
    recall = sum_spam_reviewer / 2041
    if precision + recall == 0.0:
        f1 = 0
    else:
        f1 = (2 * precision * recall) / (precision + recall)
    dataframe = pd.DataFrame(result_spammer)
    dataframe.to_csv('result spammers_YelpZip.csv', index=False, sep=',')
    # 先不将result_spammer写入文件
    # print(len(result['precision']))
    # print(len(result['index']))
    return precision, recall, f1, result


data_path = '../Data/YelpZip/'
result_path = '../output/YelpZip/'
database_path = 'G:\\DB\\YelpZip.db'
rp = data_path + 'remap_reviewer.txt'
detected_path = result_path + 'user_item_anomalyScore_0.8_0.csv'
read_result_Yelp(database_path, rp, detected_path)
