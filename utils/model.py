import time, os, joblib
import numpy as np
import tensorflow as tf

from tensorflow import keras
from utils.layer import time_mixer_block

strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
#warnings.filterwarnings(action='ignore', category=UserWarning)


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



with strategy.scope():
    def build_model(input_shape, d_dims=64, output_len=1, dropout_rate=0.2, learning_rate=0.001):
        input_layer = keras.layers.Input(shape=input_shape)

        x = keras.layers.BatchNormalization()(input_layer)
        x = keras.layers.Dense(units=d_dims, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x_res = x

        for i in range(5):
            dilation_rate = 2 ** i
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x_res)
            x = keras.layers.Dropout(dropout_rate)(x)
            x_res = x + x_res
            x_res = keras.layers.Activation('gelu')(x_res)
            x_res = keras.layers.BatchNormalization()(x_res)

        y = keras.layers.Flatten()(x_res)
        y = keras.layers.Dropout(dropout_rate)(y)
        y = keras.layers.Dense(units=30, activation='gelu')(y)
        y_res = y

        for j in range(3):
            y = time_mixer_block(input_layer=y_res, pred_len=30, dropout_rate=dropout_rate)
            y_res = y + y_res

        y = keras.layers.Dense(units=output_len, activation='linear')(y_res)
        model = keras.models.Model(inputs=input_layer, outputs=y)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=keras.losses.logcosh,
                      metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

        return model