import os
import numpy as np
import pandas as pd
from tensorflow.python.ops.numpy_ops import array

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


def get_test_case(n_case: int, n_case_iter: int) -> dict:
    test_case = {'test_case': [], 'test_case_iter': [], 'nozzle_len': [], 'nozzle_dia': [], 'venturi_dist': []}
    nozzle_len_list = [12, 24, 36]
    nozzle_dia_list = [12, 15.8, 20]
    venturi_len_list = [0, 15, 30, 45, 60]

    for i in range(n_case):
        for j in range(n_case_iter):
            nozzle_idx = i // 5
            stage_idx = i % 5
            nozzle_len_idx = nozzle_idx // 3
            nozzle_dia_idx = nozzle_idx % 3

            test_case['test_case'].append(i+1)
            test_case['test_case_iter'].append(j+1)
            test_case['nozzle_len'].append(nozzle_len_list[nozzle_len_idx])
            test_case['nozzle_dia'].append(nozzle_dia_list[nozzle_dia_idx])
            test_case['venturi_dist'].append(venturi_len_list[stage_idx])

    return test_case


def load_dataset_v2(file_path_list: list) -> pd.DataFrame:
    df_list = []
    col_name = ['time(s)', 'pressure_1(bar)', 'main_pressure(bar)', 'venturi_pressure_1(bar)', 'venturi_pressure_2(bar)',
                'venturi_pressure_3(bar)', 'venturi_pressure_4(bar)', 'venturi_pressure_5(bar)', 'pump_speed(rpm)',
                'water_temp(c)','reserved', 'outlet_flowrate(lpm)', 'inlet_flowrate(lpm)']
    extra_col_name = ['test_case', 'test_case_iter', 'nozzle_len(mm)', 'nozzle_dia(mm)', 'venturi_dist(mm)']

    col_name = col_name + extra_col_name

    test_case = get_test_case(n_case=45, n_case_iter=3)

    for file_path in tqdm(file_path_list, desc='loading dataset...'):
        file_name = os.path.basename(file_path)
        raw_data = pd.read_csv(file_path, encoding='cp949')
        data_value = raw_data.iloc[7:, :].values
        data_value = data_value.astype(np.float64)

        time_arr = np.arange(0, data_value.shape[0]) / 100
        data_value = np.insert(arr=data_value, obj=0, values=time_arr, axis=1)

        n_test_case = int(file_name[0:2])
        n_test_case_iter = int(file_name[3:5])

        sel_case_idx = (np.array(test_case['test_case']) == n_test_case) * \
                       (np.array(test_case['test_case_iter']) == n_test_case_iter)

        nozzle_len = test_case['nozzle_len'][np.argmax(sel_case_idx)]
        nozzle_dia = test_case['nozzle_dia'][np.argmax(sel_case_idx)]
        venturi_len = test_case['venturi_dist'][np.argmax(sel_case_idx)]

        test_case_arr = np.full(shape=data_value.shape[0], fill_value=n_test_case)
        test_case_iter_arr = np.full(shape=data_value.shape[0], fill_value=n_test_case_iter)
        nozzle_len_arr = np.full(shape=data_value.shape[0], fill_value=nozzle_len)
        nozzle_dia_arr = np.full(shape=data_value.shape[0], fill_value=nozzle_dia)
        venturi_len_arr = np.full(shape=data_value.shape[0], fill_value=venturi_len)

        extra_data = np.stack(arrays=(test_case_arr, test_case_iter_arr, nozzle_len_arr, nozzle_dia_arr, venturi_len_arr), axis=1)
        df_list.append(np.hstack([data_value, extra_data]))

    data_set = np.vstack(df_list)
    data_set = data_set.astype(np.float64)

    return pd.DataFrame(data=data_set, columns=col_name)


def create_lstm_dataset(data: np.array, seq_len=1, pred_distance=1, target_idx_pos=1):
    feature, target = [], []

    for i in range(data.shape[0] - seq_len - pred_distance - 1):
        feature.append(data[i:i + seq_len, 0:target_idx_pos])
        target.append(data[i + seq_len + pred_distance, target_idx_pos])

    return np.array(feature), np.array(target)
