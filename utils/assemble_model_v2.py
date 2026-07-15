"""v2 모델 — DFL(Distribution Focal Loss) 분포 헤드.

백본은 v2와 동일(BatchNorm → Conv1D 스템 → dilated causal TCN 잔차블록)하게 두고, 헤드만
단일 선형 readout에서 **유량 bin 분포 예측(DFL)**으로 교체한다. 유량 범위를 로그 등간격 bin
으로 이산화하고 softmax 분포를 예측한 뒤 기대값으로 연속 유량을 복원한다. softmax가 유량
구간에 대한 소프트 게이트 역할을 하고, 로그 bin이 저유량(1–500 lpm) 상대오차를 구조적으로
공략한다(절대오차 손실이 저유량을 홀대하는 문제 해소).

- 헤드   : GAP → Dropout → Dense(K, softmax)=분포 → ExpectationLayer=유량 스칼라
- 손실   : 하이브리드 DFL(분포 CE) + λ·LogCosh(기대값, y)  (GFL식 안정화)
- 출력   : [flow(B,1) 스칼라 추정=배포·평가용, dist(B,K) 분포=DFL 손실용]
- 참고   : Li et al. 2020 "Generalized Focal Loss"(DFL), YOLOv8.
"""
import numpy as np
import keras
from keras import ops

from utils.miscellaneous import count_divisions_by_two

_EPS = 1e-7


def log_bin_centers(v_min, v_max, num_bins):
    """로그 등간격 bin 중심값 c_i (오름차순, shape (num_bins,), float32).

    각 bin이 일정한 *비율* 구간을 담당하므로 저유량도 고유량과 동일한 상대 해상도를 얻는다.
    """
    if v_min <= 0:
        raise ValueError(f"v_min은 0보다 커야 함(로그 격자): v_min={v_min}")
    return np.exp(np.linspace(np.log(v_min), np.log(v_max), num_bins)).astype('float32')


@keras.saving.register_keras_serializable(package='dfl')
class ExpectationLayer(keras.layers.Layer):
    """유량 bin 분포(softmax)를 기대값 스칼라로 복원한다: flow = Σ_i c_i · p_i.

    bin 중심값을 비학습 가중치로 보유해 디바이스/직렬화와 함께 이동한다.
    """

    def __init__(self, v_min, v_max, num_bins, **kwargs):
        super().__init__(**kwargs)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_bins = int(num_bins)
        self._centers_np = log_bin_centers(self.v_min, self.v_max, self.num_bins)

    def build(self, input_shape):
        # 비학습 상수(bin 중심값). 모델과 함께 저장/로드된다.
        self.centers = self.add_weight(
            name='bin_centers', shape=(self.num_bins,),
            initializer=keras.initializers.Constant(self._centers_np), trainable=False)
        super().build(input_shape)

    def call(self, dist):
        # dist (B, K) · centers (K,) → (B, 1)
        return ops.sum(dist * self.centers, axis=-1, keepdims=True)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 1)

    def get_config(self):
        config = super().get_config()
        config.update({'v_min': self.v_min, 'v_max': self.v_max, 'num_bins': self.num_bins})
        return config


@keras.saving.register_keras_serializable(package='dfl')
class DFLLoss(keras.losses.Loss):
    """Distribution Focal Loss — 연속 타깃을 이웃 두 bin에 선형보간 확률질량으로 몰아주는 CE.

    타깃 y가 c_left ≤ y ≤ c_right 사이일 때:
        w_r = (y - c_left)/(c_right - c_left),  w_l = 1 - w_r
        loss = -(w_l·log p_left + w_r·log p_right)
    keras.ops만 사용해 torch 백엔드와 호환된다. searchsorted 대신 backend-안전 카운트로
    이웃 인덱스를 구한다.
    """

    def __init__(self, v_min, v_max, num_bins, name='dfl_loss', **kwargs):
        super().__init__(name=name, **kwargs)
        self.v_min = float(v_min)
        self.v_max = float(v_max)
        self.num_bins = int(num_bins)
        self._centers_np = log_bin_centers(self.v_min, self.v_max, self.num_bins)

    def call(self, y_true, y_pred):
        # y_true: 스칼라 유량 (B,) 또는 (B,1) → (B,1). y_pred: softmax 분포 (B,K).
        centers = ops.convert_to_tensor(self._centers_np)                 # (K,)
        y_pred = ops.cast(y_pred, centers.dtype)
        yt = ops.cast(ops.reshape(y_true, (-1, 1)), centers.dtype)        # (B,1)
        yt = ops.clip(yt, self._centers_np[0], self._centers_np[-1])

        # 이웃 bin 인덱스: right = (yt보다 작은 중심값 개수), [1, K-1]로 clip.
        less = ops.cast(ops.expand_dims(centers, 0) < yt, 'int32')        # (B,K)
        right = ops.clip(ops.sum(less, axis=-1), 1, self.num_bins - 1)    # (B,)
        left = right - 1

        c_left = ops.take(centers, left, axis=0)                          # (B,)
        c_right = ops.take(centers, right, axis=0)                        # (B,)
        w_r = (ops.squeeze(yt, axis=-1) - c_left) / (c_right - c_left + _EPS)  # (B,)
        w_l = 1.0 - w_r

        p_left = ops.squeeze(ops.take_along_axis(y_pred, ops.expand_dims(left, -1), axis=-1), -1)
        p_right = ops.squeeze(ops.take_along_axis(y_pred, ops.expand_dims(right, -1), axis=-1), -1)

        # 반환은 표본별 (B,) — Loss 기반 클래스가 배치 평균으로 축약.
        return -(w_l * ops.log(p_left + _EPS) + w_r * ops.log(p_right + _EPS))

    def get_config(self):
        config = super().get_config()
        config.update({'v_min': self.v_min, 'v_max': self.v_max, 'num_bins': self.num_bins})
        return config


def build_reg_model_v2(input_shape, num_bins=32, v_min=1.0, v_max=3000.0,
                       d_dims=128, dropout_rate=0.5, learning_rate=0.00242, lambda_logcosh=0.1):
    """v2 백본(Conv 스템 + dilated causal TCN) + DFL 분포 헤드.

    출력 2개: flow(스칼라 유량 추정, 배포·평가·LogCosh용), dist(유량 bin 분포, DFL용).
    실제 학습 시 v_max는 1.05×데이터 최대유량으로 설정할 것(증강 ×1.05 초과 대비). 기본 3000은
    플레이스홀더.
    """
    input_layer = keras.layers.Input(shape=input_shape)
    x = keras.layers.BatchNormalization()(input_layer)

    # === 스템: Conv1D 임베딩 (v2와 동일) ===
    x_res = keras.layers.Conv1D(filters=d_dims, kernel_size=3, padding='causal', activation='gelu')(x)

    # === 백본: dilated causal 잔차블록 (v2와 동일) ===
    for i in range(count_divisions_by_two(input_shape[0]) + 1):
        dilation_rate = 2 ** i
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                dilation_rate=dilation_rate)(x_res)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                dilation_rate=dilation_rate)(x)
        x_res = keras.layers.BatchNormalization()(x + x_res)

    # === DFL 헤드 ===
    h = keras.layers.GlobalAveragePooling1D()(x_res)
    h = keras.layers.Dropout(dropout_rate)(h)
    dist = keras.layers.Dense(units=num_bins, activation='softmax', name='dist')(h)   # 유량 bin 분포
    flow = ExpectationLayer(v_min, v_max, num_bins, name='flow')(dist)                # 기대값=유량 스칼라

    model = keras.models.Model(inputs=input_layer, outputs=[flow, dist])

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        loss={'flow': keras.losses.LogCosh(), 'dist': DFLLoss(v_min, v_max, num_bins)},
        loss_weights={'flow': lambda_logcosh, 'dist': 1.0},
        metrics={'flow': ['mean_absolute_error', 'mean_absolute_percentage_error']})

    return model
