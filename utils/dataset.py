import os
import numpy as np
import pandas as pd

from tqdm.auto import tqdm


motor_speed_750_indices = np.array([23, 19, 17, 21, 3, 5, 1, 7, 29, 25, 27, 31, 13, 15, 9, 11], dtype=np.uint8)
motor_speed_900_indices = np.array([24, 18, 20, 22, 4, 6, 8, 2, 26, 32, 28, 30, 16, 14, 10, 12], dtype=np.uint8)
hose_length_30m_indices = np.array([18, 17, 22, 21, 5, 1, 6, 2, 26, 29, 25, 30, 13, 9, 14, 10], dtype=np.uint8)
hose_length_45m_indices = np.array([24, 23, 20, 19, 3, 4, 7, 8, 27, 32, 28, 31, 16, 15, 12, 11], dtype=np.uint8)
suction_height_65cm_indices = np.array([18, 20, 19, 17, 3, 4, 1, 2, 26, 25, 27, 28, 9, 10, 12, 11], dtype=np.uint8)
suction_height_45cm_indices = np.array([24, 23, 22, 21, 5, 6, 7, 8, 29, 32, 30, 31, 16, 13, 15, 14], dtype=np.uint8)

std_val_names = ['25', '26', '27', '28', '29', '30', '31', '32']

data_root_path = 'data' + os.sep
data_file_names = os.listdir(os.path.join('data', 'ver_1'))

data_v2_train_file_list = []
data_v2_val_file_list = []


def load_dataset_v1(data_indices: list) -> pd.DataFrame():
    raw_data_set = pd.DataFrame()

    for data_index in tqdm(data_indices, desc='loading dataset...'):
        for data_file_name in data_file_names:
            raw_data = pd.read_csv(data_root_path + data_file_name)
            run_index = int(data_file_name[8:10])

            motor_speed: int = 0
            hose_length: int = 0
            suction_height: float = 0

            if np.any(run_index == motor_speed_750_indices):
                motor_speed = 750

            if np.any(run_index == motor_speed_900_indices):
                motor_speed = 900

            if np.any(run_index == hose_length_30m_indices):
                hose_length = 30

            if np.any(run_index == hose_length_45m_indices):
                hose_length = 45

            if np.any(run_index == suction_height_45cm_indices):
                suction_height = -0.45

            if np.any(run_index == suction_height_65cm_indices):
                suction_height = -0.65

            if run_index == data_index:
                motor_speed_arr = np.full(shape=raw_data.shape[0], fill_value=motor_speed)
                hose_length_arr = np.full(shape=raw_data.shape[0], fill_value=hose_length)
                suction_height_arr = np.full(shape=raw_data.shape[0], fill_value=suction_height)
                std_val_name_list = np.full(shape=raw_data.shape[0], fill_value=data_index)

                raw_data = pd.concat([raw_data, pd.DataFrame(motor_speed_arr, columns=['motor_speed(rpm)'])], axis=1)
                raw_data = pd.concat([raw_data, pd.DataFrame(hose_length_arr, columns=['hose_length(m)'])], axis=1)
                raw_data = pd.concat([raw_data, pd.DataFrame(suction_height_arr, columns=['suction_height_of_venturi(m)'])], axis=1)
                raw_data = pd.concat([raw_data, pd.DataFrame(std_val_name_list, columns=['run_name'])], axis=1)

                raw_data_set = pd.concat([raw_data_set, raw_data], axis=0)

    return raw_data_set


def load_dataset_v2(file_name_list: list) -> tuple:
    for file_name in file_name_list:
        if int(file_name[3:5]) == 3:
            data_v2_val_file_list.append(file_name)
        else:
            data_v2_train_file_list.append(file_name)

    print(data_v2_train_file_list)
    print(data_v2_val_file_list)


