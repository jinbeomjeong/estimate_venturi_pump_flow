import os
import numpy as np
import pandas as pd

from tqdm import tqdm


pipe_length_30m_indices = np.array([18, 17, 22, 21, 5, 1, 6, 2, 26, 29, 25, 30, 13, 9, 14, 10], dtype=np.uint8)
pipe_length_45m_indices = np.array([24, 23, 20, 19, 3, 4, 7, 8, 27, 32, 28, 31, 16, 15, 12, 11], dtype=np.uint8)
venturi_pump_65cm_indices = np.array([18, 20, 19, 17, 3, 4, 1, 2, 26, 25, 27, 28, 9, 10, 12, 11], dtype=np.uint8)
venturi_pump_45cm_indices = np.array([24, 23, 22, 21, 5, 6, 7, 8, 29, 32, 30, 31, 16, 13, 15, 14], dtype=np.uint8)
motor_speed_750_indices = np.array([23, 19, 17, 21, 3, 5, 1, 7, 29, 25, 27, 31, 13, 15, 9, 11], dtype=np.uint8)
motor_speed_900_indices = np.array([24, 18, 20, 22, 4, 6, 8, 2, 26, 32, 28, 30, 16, 14, 10, 12], dtype=np.uint8)

data_root_path = 'data' + os.sep
data_file_names = os.listdir('data')


def load_dataset(train_dataset_indices: np.ndarray, val_dataset_indices: np.ndarray):
    train_dataset = pd.DataFrame()
    val_dataset = pd.DataFrame()
    motor_speed = 0

    for data_file_name in tqdm(data_file_names, desc='load dataset', ncols=80):
        raw_data = pd.read_csv(data_root_path + data_file_name)
        run_index = int(data_file_name[8:10])

        if np.any(run_index == motor_speed_750_indices):
            motor_speed = 750

        if np.any(run_index == motor_speed_900_indices):
            motor_speed = 900

        motor_speed_arr = np.full(shape=raw_data.shape[0], fill_value=motor_speed)

        if np.any(run_index == train_dataset_indices):
            # raw_data = pd.concat([raw_data, pd.DataFrame(motor_speed_arr, columns=['motor_speed(rpm)'])], axis=1)
            train_dataset = pd.concat([train_dataset, raw_data], axis=0)

        if np.any(run_index == val_dataset_indices):
            # raw_data = pd.concat([raw_data, pd.DataFrame(motor_speed_arr, columns=['motor_speed(rpm)'])], axis=1)
            val_dataset = pd.concat([val_dataset, raw_data], axis=0)

    train_dataset.drop(columns=['time(sec)', 'main_pump_outlet_F(LPM)'], inplace=True)
    val_dataset.drop(columns=['time(sec)', 'main_pump_outlet_F(LPM)'], inplace=True)

    return train_dataset, val_dataset
