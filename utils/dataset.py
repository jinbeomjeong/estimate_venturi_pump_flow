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


def load_dataset_v1(data_indices: list) -> pd.DataFrame():
    # 모듈을 읽는 시점이 아니라 이 함수를 부를 때 디렉터리를 봅니다. 예전에는
    # 모듈 최상단에서 os.listdir을 하는 바람에, 해당 디렉터리가 없는 환경에서는
    # utils.dataset을 import 하는 것만으로도 FileNotFoundError가 났습니다.
    data_file_names = os.listdir(os.path.join('data', 'ver_1'))
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

FLOW_COLUMNS = ['inlet_pressure(bar)', 'outlet_pressure(bar)', 'pump_speed(rpm)', 'flowrate(lpm)']
FLOW_TEST_COLUMNS = ['pump_inlet_pressure(bar)', 'pump_outlet_pressure(bar)', 'pump_speed(rpm)',
                     'venturi_flowrate(lpm)']


def resample_by_period(raw_data: pd.DataFrame, period: float = 0.5) -> pd.DataFrame:
    """
    시간 축을 일정 간격으로 다시 뽑습니다. 각 기준 시각에 가장 가까운 샘플을 고릅니다.

    Args:
        raw_data (pd.DataFrame): 'time(sec)' 열을 포함한 원본 로그.
        period (float): 샘플 간격(초). 추론 루프 주기와 같아야 합니다.
    """
    time_arr = raw_data['time(sec)'].to_numpy()
    ref_time_arr = np.arange(0, np.round(time_arr.max(), 0), period)
    idx_list = [int(np.argmin(np.abs(time_arr - ref_time))) for ref_time in ref_time_arr]

    return raw_data.iloc[idx_list, :].reset_index(drop=True)


def split_valid_runs(raw_data: pd.DataFrame, columns: list, min_rpm: float = 700,
                     min_flowrate: float = 100) -> list:
    """
    정상 운전 구간만 남기고, 끊긴 구간을 경계로 잘라 연속 구간 리스트를 만듭니다.

    운전 범위 밖 샘플을 지운 뒤 그냥 이어붙이면 시퀀스 창이 그 틈을 가로질러
    존재하지 않는 과거를 학습하게 됩니다. 그래서 구간을 나눠 둡니다.
    """
    data_arr = raw_data[columns].to_numpy(dtype=np.float64)
    keep_mask = (data_arr[:, 2] > min_rpm) & (data_arr[:, 3] > min_flowrate)

    run_list = []
    start_idx = None

    for i, keep in enumerate(keep_mask):
        if keep and start_idx is None:
            start_idx = i

        elif not keep and start_idx is not None:
            run_list.append(data_arr[start_idx:i])
            start_idx = None

    if start_idx is not None:
        run_list.append(data_arr[start_idx:])

    return run_list


def make_windows(run_list: list, seq_len: int = 20) -> tuple:
    """
    연속 구간들을 (n_samples, seq_len, 3) 특징과 (n_samples,) 타깃으로 바꿉니다.
    창은 한 구간 안에서만 만들어지므로 서로 다른 운전을 섞지 않습니다.
    """
    feature_list, target_list = [], []

    for run in run_list:
        if run.shape[0] < seq_len:
            continue

        window = np.lib.stride_tricks.sliding_window_view(run[:, 0:3], seq_len, axis=0)
        feature_list.append(np.transpose(window, axes=(0, 2, 1)))
        target_list.append(run[seq_len - 1:, 3])

    if not feature_list:
        return np.empty((0, seq_len, 3), np.float32), np.empty((0,), np.float32)

    return (np.concatenate(feature_list).astype(np.float32),
            np.concatenate(target_list).astype(np.float32))


def load_flow_dataset(train_dir: str, test_file_list: list, seq_len: int = 20, period: float = 0.5) -> tuple:
    """
    유량 추정용 학습/평가 데이터를 읽어 시퀀스로 만듭니다.

    Args:
        train_dir (str): 헤더 없는 학습 로그(csv)가 든 디렉터리.
        test_file_list (list): 헤더가 있는 평가 로그(csv) 경로 리스트.
        seq_len (int): 입력 창 길이.
        period (float): 리샘플링 간격(초).

    Returns:
        tuple: ((train_feature, train_target), (test_feature, test_target))
    """
    train_runs = []

    for file_name in sorted(os.listdir(train_dir)):
        # 학습 로그에는 헤더 행이 없습니다. header=None을 빼면 첫 샘플이 헤더로
        # 먹히고 조용히 사라집니다.
        raw_data = pd.read_csv(os.path.join(train_dir, file_name), header=None)
        raw_data.columns = ['time(sec)'] + FLOW_COLUMNS
        train_runs += split_valid_runs(resample_by_period(raw_data, period), FLOW_COLUMNS)

    test_runs = []

    for file_path in test_file_list:
        raw_data = pd.read_csv(file_path)
        raw_data = raw_data[['time(sec)'] + FLOW_TEST_COLUMNS]
        raw_data.columns = ['time(sec)'] + FLOW_COLUMNS
        test_runs += split_valid_runs(resample_by_period(raw_data, period), FLOW_COLUMNS)

    return make_windows(train_runs, seq_len), make_windows(test_runs, seq_len)
