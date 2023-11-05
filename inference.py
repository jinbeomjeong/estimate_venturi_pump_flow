import time

import numpy as np

from utils.load_model import FlowRateInference
from utils.dataset import load_dataset


train_dataset_indices = [24, 23, 18, 20, 19, 17, 22, 21, 3, 5, 4, 1, 6, 7, 8, 2, 26, 29, 25, 27, 32, 28, 30, 31]
val_dataset_indices = [16, 13, 15, 9, 14, 10, 12, 11]
est_model = FlowRateInference()
train_feature_dataset, train_target_dataset, val_feature_dataset, val_target_dataset = load_dataset(train_dataset_indices=np.array(train_dataset_indices),
                                                                                                    val_dataset_indices=np.array(val_dataset_indices))


t0 = time.time()
for input_val, ref_output in zip(val_feature_dataset, val_target_dataset):
    # input data order: main pump inlet P, main pump outlet P, venturi pump inlet P, venturi pump outlet F
    pred_output = est_model.run_inference(input_val.reshape(1, -1))
    print(f'execute estimation time(msec): {(time.time()-t0)*1000:.1f}', f'{ref_output:.1f}', f'{pred_output[0].item():.1f}')
    t0 = time.time()
