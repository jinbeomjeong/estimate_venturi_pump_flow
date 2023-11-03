import os
import numpy as np
import pandas as pd


def load_dataset(train_dataset_indices: np.ndarray, val_dataset_indices: np.ndarray) -> (np.ndarray, np.ndarray):
    data_root_path = 'data'+os.sep
    data_file_names = os.listdir('data')

    pipe_length_arr = np.array([], dtype=np.float64)
    venturi_pump_height_arr = np.array([], dtype=np.float64)
    motor_speed_arr = np.array([], dtype=np.float64)
    main_pump_inlet_pressure_arr = np.array([], dtype=np.float64)
    main_pump_outlet_pressure_arr = np.array([], dtype=np.float64)
    venturi_pump_inlet_pressure_arr = np.array([], dtype=np.float64)
    main_pump_outlet_flowrate_arr = np.array([], dtype=np.float64)
    venturi_pump_outlet_flowrate_arr = np.array([], dtype=np.float64)

    pipe_length_35mm_indices = np.array([1], dtype=np.uint8)
    pipe_length_50mm_indices = np.array([2], dtype=np.uint8)
    venturi_pump_height_30cm_indices = np.array([3], dtype=np.uint8)
    venturi_pump_height_50cm_indices = np.array([4], dtype=np.uint8)
    motor_speed_25hz_indices = np.array([5], dtype=np.uint8)
    motor_speed_30hz_indices = np.array([6], dtype=np.uint8)

    pipe_length = 0
    venturi_pump_height = 0
    motor_speed = 0
    # load csv data and extract data
    for i, data_file_name in enumerate(data_file_names):
        run_index = int(data_file_name[8:10])
        raw_data = pd.read_csv(data_root_path+data_file_name)

        if np.any(run_index == pipe_length_35mm_indices):
            pipe_length = 35

        if np.any(run_index == pipe_length_50mm_indices):
            pipe_length = 50

        if np.any(run_index == venturi_pump_height_30cm_indices):
            venturi_pump_height = 30

        if np.any(run_index == venturi_pump_height_50cm_indices):
            venturi_pump_height = 50

        if np.any(run_index == motor_speed_25hz_indices):
            motor_speed = 25

        if np.any(run_index == motor_speed_30hz_indices):
            motor_speed = 30

        pipe_length_arr = np.append(pipe_length_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                             fill_value=pipe_length))
        venturi_pump_height_arr = np.append(venturi_pump_height_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                                             fill_value=venturi_pump_height))
        motor_speed_arr = np.append(motor_speed_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                             fill_value=motor_speed))
        main_pump_inlet_pressure_arr = np.append(main_pump_inlet_pressure_arr, raw_data['main_pump_inlet_P(bar)'].values)
        main_pump_outlet_pressure_arr = np.append(main_pump_outlet_pressure_arr, raw_data['main_pump_outlet_P(bar)'].values)
        venturi_pump_inlet_pressure_arr = np.append(venturi_pump_inlet_pressure_arr,  raw_data['venturi_pump_inlet_P(bar)'].values)
        main_pump_outlet_flowrate_arr = np.append(main_pump_outlet_flowrate_arr, raw_data['main_pump_outlet_F(LPM)'].values)
        venturi_pump_outlet_flowrate_arr = np.append(venturi_pump_outlet_flowrate_arr, raw_data['venturi_pump_outlet_F(LPM)'].values)

    feature_data = np.vstack([main_pump_inlet_pressure_arr, main_pump_outlet_pressure_arr,
                              venturi_pump_inlet_pressure_arr, main_pump_outlet_flowrate_arr]).T

    return feature_data, venturi_pump_outlet_flowrate_arr
