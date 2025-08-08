import time, os, joblib
import numpy as np

from tensorflow import keras


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


def conv_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_1x3(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv1D(filters=output_dim, kernel_size=3, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_1x5(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv1D(filters=output_dim, kernel_size=5, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def max_pool_to_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'),
        keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

class InceptionBlock(keras.layers.Layer):
    def __init__(self, output_dim_1x1=64, hidden_dim_3x3=96, output_dim_3x3=128, hidden_dim_5x5=16, output_dim_5x5=32, output_dim_max_pool=32, dropout_rate=0.2, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        self.output_dim_1x1 = output_dim_1x1
        self.hidden_dim_3x3 = hidden_dim_3x3
        self.output_dim_3x3 = output_dim_3x3
        self.hidden_dim_5x5 = hidden_dim_5x5
        self.output_dim_5x5 = output_dim_5x5
        self.output_dim_max_pool = output_dim_max_pool
        self.dropout_rate = dropout_rate

        self.conv_1x1 = conv_1x1(output_dim=self.output_dim_1x1, dropout_rate=self.dropout_rate)
        self.conv_3x3 = conv_1x3(hidden_dim=self.hidden_dim_3x3, output_dim=self.output_dim_3x3, dropout_rate=self.dropout_rate)
        self.conv_5x5 = conv_1x5(hidden_dim=self.hidden_dim_5x5, output_dim=self.output_dim_5x5, dropout_rate=self.dropout_rate)
        self.max_pool = max_pool_to_1x1(output_dim=self.output_dim_max_pool, dropout_rate=self.dropout_rate)

    def call(self, inputs_layer):
        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.max_pool(inputs_layer)

        return keras.layers.concatenate([output_layer_1, output_layer_2, output_layer_3, output_layer_4], axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5, 'output_dim_max_pool': self.output_dim_max_pool,
                       'dropout_rate': self.dropout_rate})

        return config

def build_model(input_shape=(1, 1), dropout_rate=0.2):
    input_layer = keras.layers.Input(shape=input_shape)

    inception_block_1 = InceptionBlock(output_dim_1x1=192/2, hidden_dim_3x3=96/2, output_dim_3x3=208/2, hidden_dim_5x5=16/2, output_dim_5x5=48/2, output_dim_max_pool=64/2, dropout_rate=dropout_rate)(input_layer)
    lstm_1 = keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=dropout_rate)(inception_block_1)
    y1 = keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same')(lstm_1)
    y1 = keras.layers.Dropout(0.01)(y1)
    y1 = keras.layers.Flatten()(y1)
    y1 = keras.layers.Dense(30, activation='linear', name='output_1')(y1)

    inception_block_2 = InceptionBlock(output_dim_1x1=256/2, hidden_dim_3x3=160/2, output_dim_3x3=320/2, hidden_dim_5x5=32/2, output_dim_5x5=128/2, output_dim_max_pool=128/2, dropout_rate=dropout_rate)(lstm_1)
    lstm_2 = keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=dropout_rate)(inception_block_2)
    lstm_2 = keras.layers.add([lstm_1, lstm_2])
    y2 = keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same')(lstm_2)
    y2 = keras.layers.Dropout(0.01)(y2)
    y2 = keras.layers.Flatten()(y2)
    y2 = keras.layers.Dense(30, activation='linear', name='output_2')(y2)

    inception_block_3 = InceptionBlock(output_dim_1x1=384/2, hidden_dim_3x3=192/2, output_dim_3x3=384/2, hidden_dim_5x5=48/2, output_dim_5x5=128/2, output_dim_max_pool=128/2, dropout_rate=dropout_rate)(lstm_2)
    lstm_3 = keras.layers.LSTM(units=64, return_sequences=True, recurrent_dropout=dropout_rate)(inception_block_3)
    lstm_3 = keras.layers.add([lstm_2, lstm_3])
    y3 = keras.layers.Conv1D(filters=1, kernel_size=3, strides=1, activation='relu', padding='same')(lstm_3)
    y3 = keras.layers.Dropout(0.01)(y3)
    y3 = keras.layers.Flatten()(y3)
    y3 = keras.layers.Dense(30, activation='linear', name='output_3')(y3)

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model = keras.models.Model(inputs=input_layer, outputs=[y1, y2, y3])

    model.compile(optimizer=optimizer, loss={'output_1': 'mse', 'output_2': 'mse', 'output_3':'mse' },
                  loss_weights={'output_1': 0.3, 'output_2': 0.3, 'output_3': 1.0},
                  metrics={'output_1': ['mean_absolute_error', 'mean_absolute_percentage_error'],
                           'output_2': ['mean_absolute_error', 'mean_absolute_percentage_error'],
                           'output_3': ['mean_absolute_error', 'mean_absolute_percentage_error']})

    return model
