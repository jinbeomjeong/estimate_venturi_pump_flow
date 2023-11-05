import os
import numpy as np
import pandas as pd


def load_dataset(train_dataset_indices: np.ndarray, val_dataset_indices: np.ndarray) -> (np.ndarray, np.ndarray):
    data_root_path = 'data'+os.sep
    data_file_names = os.listdir('data')

    pipe_length_arr = np.array([], dtype=np.float64)
    venturi_pump_height_arr = np.array([], dtype=np.float64)
    motor_speed_arr = np.array([], dtype=np.float64)

    main_pump_inlet_pressure_train_arr = np.array([], dtype=np.float64)
    main_pump_outlet_pressure_train_arr = np.array([], dtype=np.float64)
    venturi_pump_inlet_pressure_train_arr = np.array([], dtype=np.float64)
    main_pump_outlet_flowrate_train_arr = np.array([], dtype=np.float64)
    venturi_pump_outlet_flowrate_train_arr = np.array([], dtype=np.float64)

    main_pump_inlet_pressure_val_arr = np.array([], dtype=np.float64)
    main_pump_outlet_pressure_val_arr = np.array([], dtype=np.float64)
    venturi_pump_inlet_pressure_val_arr = np.array([], dtype=np.float64)
    main_pump_outlet_flowrate_val_arr = np.array([], dtype=np.float64)
    venturi_pump_outlet_flowrate_val_arr = np.array([], dtype=np.float64)

    pipe_length_30m_indices = np.array([18, 17, 22, 21, 5, 1, 6, 2, 26, 29, 25, 30, 13, 9, 14, 10], dtype=np.uint8)
    pipe_length_45m_indices = np.array([24, 23, 20, 19, 3, 4, 7, 8, 27, 32, 28, 31, 16, 15, 12, 11], dtype=np.uint8)
    venturi_pump_height_30cm_indices = np.array([18, 20, 19, 17, 3, 4, 1, 2, 26, 25, 27, 28, 9, 10, 12, 11], dtype=np.uint8)
    venturi_pump_height_50cm_indices = np.array([24, 23, 22, 21, 5, 6, 7, 8, 29, 32, 30, 31, 16, 13, 15, 14], dtype=np.uint8)
    motor_speed_25hz_indices = np.array([23, 19, 17, 21, 3, 5, 1, 7, 29, 25, 27, 31, 13, 15, 9, 11], dtype=np.uint8)
    motor_speed_30hz_indices = np.array([24, 18, 20, 22, 4, 6, 8, 2, 26, 32, 28, 30, 16, 14, 10, 12], dtype=np.uint8)

    pipe_length = 0  # unit: meter
    venturi_pump_height = 0  # unit: cent-meter
    motor_speed = 0  # unit: hz

    for i, data_file_name in enumerate(data_file_names):  # load csv data and extract data
        run_index = int(data_file_name[8:10])

        if np.any(run_index == motor_speed_25hz_indices):
            motor_speed = 25

        if np.any(run_index == motor_speed_30hz_indices):
            motor_speed = 30

        if motor_speed == 25:
            raw_data = pd.read_csv(data_root_path+data_file_name)

            if np.any(run_index == pipe_length_30m_indices):
                pipe_length = 30

            if np.any(run_index == pipe_length_45m_indices):
                pipe_length = 45

            if np.any(run_index == venturi_pump_height_30cm_indices):
                venturi_pump_height = 30

            if np.any(run_index == venturi_pump_height_50cm_indices):
                venturi_pump_height = 50

            pipe_length_arr = np.append(pipe_length_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                                 fill_value=pipe_length))
            venturi_pump_height_arr = np.append(venturi_pump_height_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                                                 fill_value=venturi_pump_height))
            motor_speed_arr = np.append(motor_speed_arr, np.full(shape=len(raw_data['main_pump_inlet_P(bar)'].values),
                                                                 fill_value=motor_speed))

            if np.any(run_index == train_dataset_indices):
                main_pump_inlet_pressure_train_arr = np.append(main_pump_inlet_pressure_train_arr,
                                                               raw_data['main_pump_inlet_P(bar)'].values)
                main_pump_outlet_pressure_train_arr = np.append(main_pump_outlet_pressure_train_arr,
                                                                raw_data['main_pump_outlet_P(bar)'].values)
                venturi_pump_inlet_pressure_train_arr = np.append(venturi_pump_inlet_pressure_train_arr,
                                                                  raw_data['venturi_pump_inlet_P(bar)'].values)
                main_pump_outlet_flowrate_train_arr = np.append(main_pump_outlet_flowrate_train_arr,
                                                                raw_data['main_pump_outlet_F(LPM)'].values)
                venturi_pump_outlet_flowrate_train_arr = np.append(venturi_pump_outlet_flowrate_train_arr,
                                                                   raw_data['venturi_pump_outlet_F(LPM)'].values)

            if np.any(run_index == val_dataset_indices):
                main_pump_inlet_pressure_val_arr = np.append(main_pump_inlet_pressure_val_arr,
                                                             raw_data['main_pump_inlet_P(bar)'].values)
                main_pump_outlet_pressure_val_arr = np.append(main_pump_outlet_pressure_val_arr,
                                                              raw_data['main_pump_outlet_P(bar)'].values)
                venturi_pump_inlet_pressure_val_arr = np.append(venturi_pump_inlet_pressure_val_arr,
                                                                raw_data['venturi_pump_inlet_P(bar)'].values)
                main_pump_outlet_flowrate_val_arr = np.append(main_pump_outlet_flowrate_val_arr,
                                                              raw_data['main_pump_outlet_F(LPM)'].values)
                venturi_pump_outlet_flowrate_val_arr = np.append(venturi_pump_outlet_flowrate_val_arr,
                                                                 raw_data['venturi_pump_outlet_F(LPM)'].values)

    train_feature_dataset = np.vstack([main_pump_inlet_pressure_train_arr, main_pump_outlet_pressure_train_arr,
                                       venturi_pump_inlet_pressure_train_arr, main_pump_outlet_flowrate_train_arr]).T
    val_feature_dataset = np.vstack([main_pump_inlet_pressure_val_arr, main_pump_outlet_pressure_val_arr,
                                     venturi_pump_inlet_pressure_val_arr, main_pump_outlet_flowrate_val_arr]).T

    return (train_feature_dataset, venturi_pump_outlet_flowrate_train_arr,
            val_feature_dataset, venturi_pump_outlet_flowrate_val_arr)
