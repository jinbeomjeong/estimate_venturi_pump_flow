import numpy as np
import tensorflow as tf
import keras

from utils.sub_layer import conv_1d_1x1, conv_1d_1x3, conv_1d_1x5, conv_1d_1x7, max_pool_1d_to_1x1
from utils.sub_layer import conv_2d_1x1, conv_2d_1x3, conv_2d_1x5, max_pool_2d_to_1x1


def gelu_approximate(x):
    return tf.nn.gelu(x, approximate=True)

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        """
        포지셔널 인코딩 레이어를 초기화합니다.

        Args:
            position (int): 시퀀스의 최대 길이 (최대 문장 길이)
            d_model (int): 임베딩 벡터의 차원
        """
        super(PositionalEncoding, self).__init__(**kwargs) # **kwargs 전달
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        """
        각도 계산을 위한 내부 함수
        """
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return position * angles

    def positional_encoding(self, position, d_model):
        """
        포지셔널 인코딩 행렬을 생성합니다.
        """
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis],
                                     np.arange(d_model)[np.newaxis, :],
                                     d_model)

        # 짝수 인덱스에는 사인 함수 적용
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        # 홀수 인덱스에는 코사인 함수 적용
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, ...]
        return tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        """
        레이어의 정방향 계산을 수행합니다.
        입력 텐서에 포지셔널 인코딩을 더합니다.
        """
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position,
                       'd_model': self.d_model})

        return config


class InceptionBlock1D(keras.layers.Layer):
    def __init__(self, output_dim_1x1=64, hidden_dim_3x3=96, output_dim_3x3=128, hidden_dim_5x5=16, output_dim_5x5=32,
                 hidden_dim_7x7=24, output_dim_7x7=32, output_dim_max_pool=32, dropout_rate=0.2, **kwargs):
        super(InceptionBlock1D, self).__init__(**kwargs)
        self.output_dim_1x1 = output_dim_1x1
        self.hidden_dim_3x3 = hidden_dim_3x3
        self.output_dim_3x3 = output_dim_3x3
        self.hidden_dim_5x5 = hidden_dim_5x5
        self.output_dim_5x5 = output_dim_5x5
        self.hidden_dim_7x7 = hidden_dim_7x7
        self.output_dim_7x7 = output_dim_7x7
        self.output_dim_max_pool = output_dim_max_pool
        self.dropout_rate = dropout_rate

        self.conv_1x1 = conv_1d_1x1(output_dim=self.output_dim_1x1, dropout_rate=self.dropout_rate)
        self.conv_3x3 = conv_1d_1x3(hidden_dim=self.hidden_dim_3x3, output_dim=self.output_dim_3x3, dropout_rate=self.dropout_rate)
        self.conv_5x5 = conv_1d_1x5(hidden_dim=self.hidden_dim_5x5, output_dim=self.output_dim_5x5, dropout_rate=self.dropout_rate)
        self.conv_7x7 = conv_1d_1x7(hidden_dim=self.hidden_dim_7x7, output_dim=self.output_dim_7x7)
        self.max_pool = max_pool_1d_to_1x1(output_dim=self.output_dim_max_pool, dropout_rate=self.dropout_rate)

    def call(self, inputs_layer):
        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.conv_7x7(inputs_layer)
        output_layer_5 = self.max_pool(inputs_layer)

        return keras.layers.concatenate(inputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4, output_layer_5], axis=2)

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5,
                       'output_dim_max_pool': self.output_dim_max_pool, 'dropout_rate': self.dropout_rate})

        return config


class TransformerEncoderBlock(keras.layers.Layer):
    def __init__(self, head_size, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.head_size = head_size
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.attention = keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)
        self.dropout1 = keras.layers.Dropout(dropout_rate)
        self.norm1 = keras.layers.LayerNormalization(epsilon=1e-6)

        self.ffn_dense1 = keras.layers.Dense(ff_dim, activation="gelu")
        self.ffn_dense2 = keras.layers.Dense(head_size, activation="linear")
        self.dropout2 = keras.layers.Dropout(dropout_rate)
        self.norm2 = keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        attention_output = self.dropout1(attention_output)
        out1 = inputs + attention_output
        norm_out1 = self.norm1(out1)

        ffn_output = self.ffn_dense1(norm_out1)
        ffn_output = self.ffn_dense2(ffn_output)
        ffn_output = self.dropout2(ffn_output)
        ffn_output = norm_out1 + ffn_output
        ffn_output = self.norm2(ffn_output)

        return ffn_output

    def get_config(self):
        config = super().get_config()
        config.update({'head_size': self.head_size,
                       'num_heads': self.num_heads,
                       'ff_dim': self.ff_dim,
                       'dropout_rate': self.dropout_rate})

        return config

class InceptionBlock2D(keras.layers.Layer):
    def __init__(self, output_dim_1x1=64, hidden_dim_3x3=96, output_dim_3x3=128, hidden_dim_5x5=16, output_dim_5x5=32, output_dim_max_pool=32,
                 dropout_rate=0.2, **kwargs):
        super(InceptionBlock2D, self).__init__(**kwargs)
        self.output_dim_1x1 = output_dim_1x1
        self.hidden_dim_3x3 = hidden_dim_3x3
        self.output_dim_3x3 = output_dim_3x3
        self.hidden_dim_5x5 = hidden_dim_5x5
        self.output_dim_5x5 = output_dim_5x5
        self.output_dim_max_pool = output_dim_max_pool
        self.dropout_rate = dropout_rate

        self.conv_1x1 = conv_2d_1x1(output_dim=self.output_dim_1x1, dropout_rate=self.dropout_rate)
        self.conv_3x3 = conv_2d_1x3(hidden_dim=self.hidden_dim_3x3, output_dim=self.output_dim_3x3, dropout_rate=self.dropout_rate)
        self.conv_5x5 = conv_2d_1x5(hidden_dim=self.hidden_dim_5x5, output_dim=self.output_dim_5x5, dropout_rate=self.dropout_rate)
        self.max_pool = max_pool_2d_to_1x1(output_dim=self.output_dim_max_pool, dropout_rate=self.dropout_rate)

    def call(self, inputs_layer):

        output_layer_1 = self.conv_1x1(inputs_layer)
        output_layer_2 = self.conv_3x3(inputs_layer)
        output_layer_3 = self.conv_5x5(inputs_layer)
        output_layer_4 = self.max_pool(inputs_layer)

        result = keras.layers.concatenate(inputs=[output_layer_1, output_layer_2, output_layer_3, output_layer_4], axis=1)

        return result

    def get_config(self):
        config = super().get_config()
        config.update({'output_dim_1x1': self.output_dim_1x1, 'hidden_dim_3x3': self.hidden_dim_3x3, 'output_dim_3x3': self.output_dim_3x3,
                       'hidden_dim_5x5': self.hidden_dim_5x5, 'output_dim_5x5': self.output_dim_5x5, 'output_dim_max_pool': self.output_dim_max_pool,
                       'dropout_rate': self.dropout_rate})

        return config

#@keras.saving.register_keras_serializable()
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

    def get_config(self):
        config = super(DecompositionLayer, self).get_config()
        config.update({"kernel_size": self.kernel_size})

        return config


#@keras.saving.register_keras_serializable()
class ScalingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale_vector = None
        self.activation = keras.layers.Activation('gelu')

    def build(self, input_shape):
        self.scale_vector = self.add_weight(shape=(input_shape[-1],), initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        y = inputs * self.scale_vector

        return self.activation(y)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ScalingLayer, self).get_config()

        return config


class FeatureWiseScalingLayer(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.layers.Activation(gelu_approximate)
        self.scaling_vector = None

    def build(self, input_shape):
        feature_dim = input_shape[-1]
        self.scaling_vector = self.add_weight(shape=(feature_dim,), initializer='ones', trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        y = inputs*self.scaling_vector
        y = self.activation(y)

        return y

    def compute_output_shape(self, input_shape):
        return input_shape


class ChannelSelect(keras.layers.Layer):
    """
    입력 텐서에서 지정한 채널만 골라냅니다.

    로깅/추론 버퍼는 (시간, 3) 모양을 그대로 유지하되, 모델이 실제로 보는 채널은
    여기서 확정합니다. 정규화 레이어가 알아서 무시해주길 기대하는 대신
    구조로 못 박는 쪽이 안전합니다.

    Args:
        channels (tuple): 사용할 채널 인덱스. 예) (0, 1) - 압력 2채널만 사용
    """
    def __init__(self, channels=(0, 1), **kwargs):
        super().__init__(**kwargs)
        self.channels = tuple(int(c) for c in channels)

    def call(self, inputs):
        return tf.gather(inputs, list(self.channels), axis=-1)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[:-1]) + (len(self.channels),)

    def get_config(self):
        config = super().get_config()
        config.update({'channels': list(self.channels)})

        return config


class MultiScaleSmoothing(keras.layers.Layer):
    """
    윈도우를 여러 길이로 평균 내어 한 벡터로 합칩니다.

    유량 추정에서 20스텝 창이 실제로 하는 일은 '동특성 파악'이 아니라
    '센서 잡음 제거'입니다(마지막 1샘플만 쓰면 MAPE가 약 0.4%p 나빠짐).
    그래서 팽창 합성곱이 평균을 스스로 배우도록 두지 않고, 여러 구간의
    평균을 직접 만들어 넘겨줍니다.

    Args:
        spans (tuple): 평균을 낼 뒤쪽 구간 길이들. 예) (1, 3, 5, 10, 20)

    출력 shape: (batch, len(spans) * channels)
    """
    def __init__(self, spans=(1, 3, 5, 10, 20), **kwargs):
        super().__init__(**kwargs)
        self.spans = tuple(spans)

    def call(self, inputs):
        pooled = [tf.reduce_mean(inputs[:, -span:, :], axis=1) for span in self.spans]

        return keras.layers.concatenate(pooled, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], len(self.spans) * input_shape[-1]

    def get_config(self):
        config = super().get_config()
        config.update({'spans': list(self.spans)})

        return config


class ChannelGate(keras.layers.Layer):
    """
    입력 채널마다 학습 가능한 스칼라를 곱합니다. 가중치에 L2 벌점이 걸려 있어
    '도움이 될 때만' 채널을 쓰게 됩니다.

    흡입측 압력은 한 세션 안에서는 유량과 잘 맞지만, 세션이 바뀌면 회귀계수의
    부호까지 뒤집힙니다(학습 -7719, 검증 -244, 테스트 +1177). 그대로 두면
    모델이 학습 세션에서만 통하는 관계를 외웁니다. 이 게이트가 그 의존도에
    비용을 매깁니다.

    Args:
        l2 (float): 게이트 가중치에 걸 L2 벌점.
        init (tuple): 채널별 게이트 초깃값. 불안정한 채널은 작게 시작시킵니다.
    """
    def __init__(self, l2=1e-2, init=(0.3, 1.0), **kwargs):
        super().__init__(**kwargs)
        self.l2 = l2
        self.init = tuple(init)

    def build(self, input_shape):
        # init 길이가 채널 수와 다르면 x * gate 가 브로드캐스팅되면서 채널 수가
        # 조용히 늘어납니다. 그러면 그래프를 만들 때와 실제로 부를 때의 모양이
        # 달라져 한참 뒤에야 엉뚱한 곳에서 터집니다. 여기서 바로 막습니다.
        if len(self.init) != input_shape[-1]:
            raise ValueError(f'ChannelGate init has {len(self.init)} values but the input has '
                             f'{input_shape[-1]} channels; give one initial value per channel.')

        self.gate = self.add_weight(shape=(input_shape[-1],), name='gate',
                                    initializer=keras.initializers.Constant(np.asarray(self.init, dtype=np.float32)),
                                    regularizer=keras.regularizers.L2(self.l2), trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs * self.gate

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'l2': self.l2, 'init': list(self.init)})

        return config


class ScaledResidual(keras.layers.Layer):
    """
    y = trunk + alpha * correction  (alpha는 0에서 시작하는 학습 가능한 스칼라)

    선형 줄기(trunk)가 처음부터 예측을 책임지고, 비선형 보정(correction)은
    alpha가 자라는 만큼만 개입합니다. 학습 초반이 안정되고, 비선형 항이
    예측을 얼마나 끌고 갈 수 있는지도 한 숫자로 확인할 수 있습니다.
    """
    def __init__(self, init=0.0, **kwargs):
        super().__init__(**kwargs)
        self.init = init

    def build(self, input_shape):
        self.alpha = self.add_weight(shape=(1,), name='alpha', initializer=keras.initializers.Constant(self.init),
                                     trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        trunk, correction = inputs

        return trunk + (self.alpha * correction)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super().get_config()
        config.update({'init': self.init})

        return config
