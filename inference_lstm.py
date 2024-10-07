import os, argparse, logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import pandas as pd

from tensorflow import keras
from utils.dataset import create_lstm_dataset
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

parser = argparse.ArgumentParser()
parser.add_argument("--file_path", type=str, default='')
args = parser.parse_args()
logger.info("initialized")

data = pd.read_csv('data.csv')
feature_data = data[['pressure_1', 'pressure_2']]
target_data = data['flowrate']

data_column_list = data.columns
logger.info("Data loaded")

target_idx = 0
exist_flowrate = np.any('flowrate' == data_column_list)

if exist_flowrate:
    feature_col_name_list = data_column_list[0:2]
    target_idx = 4
else:
    feature_col_name_list = data_column_list
    target_idx = -1

extra_feature = pd.DataFrame()

for col_name in feature_col_name_list:
    extra_feature['grad_' + col_name] = np.gradient(data[col_name])

dataset = pd.concat([feature_data, extra_feature, target_data], axis=1)

seq_len = 30
pred_distance = 0

feature_lstm, target = create_lstm_dataset(dataset.to_numpy(), seq_len=seq_len, pred_distance=pred_distance, target_idx_pos=target_idx)
logger.info("preprocessed data")

model = keras.models.load_model('lstm_model.keras')
logger.info("loaded model")

pred = np.squeeze(model.predict(feature_lstm, verbose=0))
logger.info("estimation completed")

if exist_flowrate:
    r2_score_value = r2_score(target, pred)
    mae_value = mean_absolute_error(target, pred)
    mape_value = mean_absolute_percentage_error(target, pred)

    logger.info(f'r2 score: {r2_score_value:.2f}, mean absolute error(LPM): {mae_value:.2f}, mean absolute percentage error(%): {mape_value:.2f}')

result = pd.DataFrame(pred, columns=['estimated_flowrate'])

data_sliced = data.iloc[seq_len-1:]
data_sliced.reset_index(drop=True, inplace=True)

result_data = pd.concat([data_sliced, result], axis=1)
result_data.to_csv(path_or_buf='result.csv', index=False)
logger.info("result file is saved")
