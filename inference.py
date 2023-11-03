import time
from utils.load_model import FlowRateInference
from utils.dataset import load_dataset

est_model = FlowRateInference()
feature_data, target_data = load_dataset(train_dataset_indices=[1, 2, 3], val_dataset_indices=[5])


t0 = time.time()
for input_val, ref_output in zip(feature_data, target_data):
    pred_output = est_model.run_inference(input_val.reshape(1, -1))
    print(f'execute estimation time(msec): {(time.time()-t0)*1000:.1f}', f'{ref_output:.1f}', f'{pred_output.item():.1f}')
    t0 = time.time()
