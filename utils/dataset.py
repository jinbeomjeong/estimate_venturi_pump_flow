import os
import numpy as np
import pandas as pd


def load_dataset() -> (np.ndarray, np.ndarray):
    data_root_path = 'data'+os.sep
    data_paths = os.listdir('data')

    motor_speed = np.array([], dtype=np.float64)
    pressure_1 = np.array([], dtype=np.float64)
    pressure_2 = np.array([], dtype=np.float64)
    pressure_3 = np.array([], dtype=np.float64)
    flow_rate_1 = np.array([], dtype=np.float64)
    flow_rate_2 = np.array([], dtype=np.float64)

    # load csv data and extract data
    for i, path in enumerate(data_paths):
        raw_data = pd.read_csv(data_root_path+path)
        motor_speed = np.append(motor_speed, np.full(shape=len(raw_data['pressure-1(bar)'].values), fill_value=path[0:path.find('-')-3], dtype=np.float64))
        pressure_1 = np.append(pressure_1, raw_data['pressure-1(bar)'].values)
        pressure_2 = np.append(pressure_2, raw_data['pressure-2(bar)'].values)
        pressure_3 = np.append(pressure_3, raw_data['pressure-3(bar)'].values)
        flow_rate_1 = np.append(flow_rate_1, raw_data['flow_rate_1(LPM)'].values)
        flow_rate_2 = np.append(flow_rate_2, raw_data['flow_rate_2(LPM)'].values)

    feature_data = np.vstack([motor_speed, pressure_1, pressure_2, pressure_3, flow_rate_1]).T

    return feature_data, flow_rate_2
