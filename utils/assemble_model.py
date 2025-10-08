import tensorflow as tf
from tensorflow import keras
from utils.layer import InceptionBlock1D
from utils.model import time_mixer_block
from utils.metric import smape
from utils.miscellaneous import count_divisions_by_two


strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


def build_model(input_shape=(1, 1), dropout_rate=0.2):
    input_layer = keras.layers.Input(shape=input_shape, name='input_layer')
    y1 = input_layer

    for i in range(3):
        y1_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=64, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y1)
        y1_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y1_1)
        y1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y1_1)

    y1 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y1)
    y1 = keras.layers.Dropout(dropout_rate)(y1)
    y1 = keras.layers.Dense(units=1, activation='linear', name='output_1')(y1)

    y2 = keras.layers.concatenate(inputs=[y1, input_layer], axis=2)

    for i in range(3):
        y2_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y2)
        y2_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y2_1)
        y2 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y2_1)

    y2 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y2)
    y2 = keras.layers.Dropout(dropout_rate)(y2)
    y2 = keras.layers.Dense(units=1, activation='linear')(y2)
    y2_add = keras.layers.add(inputs=[y2, y1], name='output_2')

    y3 = keras.layers.concatenate(unputs=[y2_add, input_layer], axis=2)

    for i in range(3):
        y3_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y3)
        y3_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y3_1)
        y3 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y3_1)

    y3 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y3)
    y3 = keras.layers.Dropout(dropout_rate)(y3)
    y3 = keras.layers.Dense(units=1, activation='linear')(y3)
    y3_add = keras.layers.add(inputs=[y3, y2_add], name='output_3')

    y4 = keras.layers.concatenate(inputs=[y3_add, input_layer], axis=2)

    for j in range(3):
        y4_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y4)
        y4_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y4_1)
        y4 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y4_1)

    y4 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y4)
    y4 = keras.layers.Dropout(dropout_rate)(y4)
    y4 = keras.layers.Dense(units=1, activation='linear')(y4)

    y4_add = keras.layers.add([y4, y3_add], name='output_4')

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model = keras.models.Model(inputs=input_layer, outputs=[y1, y2_add, y3_add, y4_add])
    model.compile(optimizer=optimizer, loss={'output_1': 'mse', 'output_2': 'mse', 'output_3': 'mse', 'output_4': 'mse'},
                  loss_weights={'output_1': 1.0, 'output_2': 1.0, 'output_3': 1.0, 'output_4': 1.0},
                  metrics={'output_1': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_2': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_3': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_4': ['mean_absolute_error', 'mean_absolute_percentage_error', smape]})
    return model



with strategy.scope():
    def build_time_mixer_model(input_shape, d_dims=64, output_len=1, dropout_rate=0.2, learning_rate=0.001):
        input_layer = keras.layers.Input(shape=input_shape)

        #x = keras.layers.BatchNormalization()(input_layer)
        x = keras.ops.expand_dims(input_layer, axis=2)
        x = keras.layers.Dense(units=d_dims, activation='gelu')(x)
        x = keras.layers.BatchNormalization()(x)
        x_res = x

        for i in range(5):
            dilation_rate = 2 ** i
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x_res)
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x_res = keras.layers.Activation('gelu')(x + x_res)
            x_res = keras.layers.BatchNormalization()(x_res)

        y = keras.layers.Flatten()(x_res)
        y = keras.layers.Dropout(dropout_rate)(y)
        y = keras.layers.Dense(units=input_shape[0]*3, activation='gelu')(y)
        y_res = y

        for j in range(3):
            y = time_mixer_block(input_layer=y_res, pred_len=input_shape[0]*3, dropout_rate=dropout_rate)
            y_res = y + y_res

        y = keras.layers.Dense(units=output_len, activation='linear')(y_res)
        model = keras.models.Model(inputs=input_layer, outputs=y)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=keras.losses.logcosh,
                      metrics=['mean_absolute_error', 'mean_absolute_percentage_error', smape])

        return model


with strategy.scope():
    def build_model_v1(input_shape, d_dims=64, output_len=1, dropout_rate=0.2, learning_rate=0.001):
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
            x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                    dilation_rate=dilation_rate)(x)
            x = keras.layers.Dropout(dropout_rate)(x)
            x_res = keras.layers.Dropout(dropout_rate)(x_res)

            x_res = keras.layers.BatchNormalization()(x+x_res)
            x_res = keras.layers.Activation('gelu')(x_res)
            x_res = keras.layers.BatchNormalization()(x_res)

        y = keras.layers.Dense(units=16, activation='gelu')(x_res)
        y_res = keras.layers.LayerNormalization()(y)

        for j in range(3):
            y = keras.layers.LSTM(units=16, return_sequences=True, recurrent_dropout=dropout_rate, dropout=dropout_rate)(y_res)
            y_res = y + y_res
            y_res = keras.layers.LayerNormalization()(y_res)

        #y = keras.ops.expand_dims(y_res, axis=3)

        #y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_last')(y)
        #y = keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='valid', data_format='channels_last')(y)

        y = keras.layers.Flatten()(y)

        y = keras.layers.LayerNormalization()(y)
        #y = keras.layers.Dropout(dropout_rate)(y)
        #y = keras.layers.Dense(units=y.shape[1], activation='gelu')(y)

        #y = keras.layers.LayerNormalization()(y)
        y = keras.ops.mean(y, axis=1)

        model = keras.models.Model(inputs=input_layer, outputs=y)

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=keras.losses.logcosh,
                      metrics=['mean_absolute_error', 'mean_absolute_percentage_error', smape])

        return model
