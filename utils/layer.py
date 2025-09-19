import tensorflow as tf

from tensorflow import keras
from utils.miscellaneous import count_divisions_by_two


def conv_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_1x3(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv1D(filters=output_dim, kernel_size=3, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def conv_1x5(hidden_dim=4, output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.Conv1D(filters=hidden_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Conv1D(filters=output_dim, kernel_size=5, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
        keras.layers.Dropout(dropout_rate)
    ])
    return model

def max_pool_to_1x1(output_dim=8, dropout_rate=0.2):
    model = keras.Sequential([
        keras.layers.MaxPooling1D(pool_size=3, strides=1, padding='same'),
        keras.layers.Conv1D(filters=output_dim, kernel_size=1, activation=None, padding='same'),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('gelu'),
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

    def build(self, input_shape):
        super(InceptionBlock, self).build(input_shape)

    def call(self, inputs_layer):
        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.max_pool(inputs_layer)

        return keras.layers.concatenate(inputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4], axis=2)

    def get_config(self):
        config = super(InceptionBlock, self).get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5, 'output_dim_max_pool': self.output_dim_max_pool,
                       'dropout_rate': self.dropout_rate})

        return config


def differencing_with_padding(x):
    # 1. 차분 계산 (결과 shape: (batch, 29, 1))
    diff = x[:, 1:, :] - x[:, :-1, :]

    # 2. 패딩용 텐서 생성 (shape: (batch, 1, 1))
    # 입력 x와 동일한 배치, 피처 크기를 갖는 0 텐서를 생성
    padding = keras.ops.zeros_like(x[:, :1, :])

    # 3. 패딩과 차분 결과를 시간 축(axis=1)을 기준으로 합침
    return keras.layers.Concatenate(axis=1)([padding, diff])


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    """트랜스포머 인코더 블록"""
    # Multi-Head Self-Attention
    x = keras.layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(inputs, inputs)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + inputs) # Add & Norm

    # Position-wise Feed-Forward Network
    ff_out = keras.layers.Dense(units=ff_dim, activation="gelu")(x)
    ff_out = keras.layers.Dropout(dropout)(ff_out)
    ff_out = keras.layers.Dense(units=head_size)(ff_out)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + ff_out) # Add & Norm

    return x

class DecompositionLayer(keras.layers.Layer):
    """
    이동 평균을 사용하여 시계열을 추세와 계절성 성분으로 분해합니다.
    """
    def __init__(self, kernel_size, **kwargs):
        super(DecompositionLayer, self).__init__(**kwargs)
        self.kernel_size = kernel_size
        self.avg = keras.layers.AvgPool1D(pool_size=kernel_size, strides=1, padding='same')

    def call(self, x):
        trend = self.avg(x)
        seasonal = x - trend
        return seasonal, trend

    # 💡 아래 메서드를 추가하여 오류를 해결합니다.
    def get_config(self):
        """레이어의 설정을 직렬화(serialize)하기 위해 호출됩니다."""
        config = super(DecompositionLayer, self).get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


def time_mixer_block(input_layer, pred_len=1, dropout_rate=0.2):
    input_raw = keras.ops.expand_dims(input_layer, axis=2)

    multi_scale_input_list = [input_raw]

    for i in range(count_divisions_by_two(input_raw.shape[1])-1):
        i = (i*2)+2
        avg_layer = keras.layers.AveragePooling1D(pool_size=i, strides=i, padding='valid')(input_raw)
        #max_layer = keras.layers.MaxPooling1D(pool_size=i, strides=i, padding='valid')(input_raw)
        multi_scale_input_list.append(avg_layer)
        #multi_scale_input_list.append(max_layer)

    seasonal_list = []
    trend_list = []

    for multi_scale_input_layer in multi_scale_input_list:
        seasonal, trend = DecompositionLayer(kernel_size=3)(multi_scale_input_layer)

        seasonal = keras.ops.squeeze(seasonal, axis=2)
        seasonal_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(seasonal)
        seasonal_output = keras.layers.Dropout(dropout_rate)(seasonal_output)
        seasonal_list.append(seasonal_output)

        trend = keras.ops.squeeze(trend, axis=2)
        trend_output = keras.layers.Dense(units=multi_scale_input_layer.shape[1], activation='linear')(trend)
        trend_output = keras.layers.Dropout(dropout_rate)(trend_output)
        trend_list.append(trend_output)

        #output_list.append(keras.layers.Add()([seasonal_output, trend_output]))

    output_1 = seasonal_list[0]
    seasonal_mix_list = [output_1]

    for i in range(len(seasonal_list)-1):
        output_1 = keras.layers.Dense(units=seasonal_list[i+1].shape[1], activation='linear')(output_1)
        output_1 = keras.layers.LayerNormalization()(output_1)
        output_1 = keras.layers.Dropout(dropout_rate)(output_1)
        output_1 = keras.layers.Activation('gelu')(output_1) #gelu
        output_1 = keras.layers.add([output_1, seasonal_list[i+1]])
        seasonal_mix_list.append(output_1)

    trend_list.reverse()
    output_2 = trend_list[0]
    trend_mix_list = [output_2]

    for i in range(len(trend_list)-1):
        output_2 = keras.layers.Dense(units=trend_list[i+1].shape[1], activation='linear')(output_2)
        output_2 = keras.layers.LayerNormalization()(output_2)
        output_2 = keras.layers.Dropout(dropout_rate)(output_2)
        output_2 = keras.layers.Activation('gelu')(output_2) #gelu
        output_2 = keras.layers.add([output_2, trend_list[i+1]])
        trend_mix_list.append(output_2)

    trend_mix_list.reverse()

    mix_output_list = []
    hidden_units = 128

    for seasonal_mix_layer, trend_mix_layer in zip(seasonal_mix_list, trend_mix_list):
        mix_output = seasonal_mix_layer+trend_mix_layer
        mix_output = keras.layers.Dense(units=hidden_units, activation='linear')(mix_output)
        mix_output = keras.layers.LayerNormalization()(mix_output)
        mix_output = keras.layers.Dropout(dropout_rate)(mix_output)
        mix_output = keras.layers.Dense(units=pred_len, activation='gelu')(mix_output) #gelu
        mix_output_list.append(mix_output)

    return keras.layers.add(mix_output_list)