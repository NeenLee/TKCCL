# NingLi 2023/12/1
# This is a main function

from AnalyzeDataset.dataset import sample_data
from AnalyzeDataset.dataset import database

if __name__ == '__main__':
    test = sample_data()
    # read_label_data(file_path=test.file_path)
    # proportion_spammer(file_path=test.file_path)
    # rewrite(file_path=test.file_path, file_write_path=test.file_write_path)

    table_name_years = 2010

    # database(file_path=test.file_path, databased_path=test.database_path, table_name=test.table_name_all)
    for i in range(0, 5):
        table_name = f'Cell_Phones_and_Accessories_{table_name_years}'
        table_file_path = f'G:\Dataset\Cell_Phones_and_Accessories\Cell_Phones_and_Accessories_{table_name_years}.json'
        database(file_path=table_file_path, databased_path=test.database_path, table_name=table_name)
        print(table_file_path, test.database_path, table_name)
        table_name_years += 1
