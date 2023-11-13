import time, os, joblib, warnings
import numpy as np


warnings.filterwarnings(action='ignore', category=UserWarning)


class FlowRateInference:
    def __init__(self):
        # load saved model
        self.t0 = time.time()
        self.pred_output = np.zeros(shape=1, dtype=np.float64)
        self.model = joblib.load('saved_model' + os.sep + 'basic_lgb_model.pkl')
        print(f"model load time(sec): {(time.time() - self.t0):.1f}")

    def run_inference(self, input_data: np.ndarray) -> np.ndarray:
        self.pred_output[0] = self.model.predict(input_data, num_iteration=self.model._best_iteration)

        return self.pred_output
