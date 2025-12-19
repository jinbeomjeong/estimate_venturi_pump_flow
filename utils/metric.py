import tensorflow as tf
import keras
from tensorflow.keras import backend as K


def smape(y_true, y_pred):
    """
    Keras용 SMAPE(Symmetric Mean Absolute Percentage Error) 메트릭 함수

    수식: 100/n * Σ(|y_pred - y_true| / ((|y_true| + |y_pred|)/2))
    """
    # 0으로 나누는 것을 방지하기 위해 작은 epsilon 값을 더해줍니다.

    # 분자: 예측값과 실제값의 차이의 절댓값
    numerator = K.abs(y_pred - y_true)

    # 분모: 실제값과 예측값의 절댓값의 합을 2로 나눔
    denominator = (K.abs(y_true) + K.abs(y_pred)) / 2 + K.epsilon()

    # SMAPE 계산
    # 각 데이터 포인트에 대한 백분율 오류를 계산하고 평균을 냅니다.
    percent_error = (numerator / denominator) * 100

    return K.mean(percent_error)


def r_squared(y_true, y_pred):
    """
    R-squared 계산 함수
    """
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - SS_res/(SS_tot + K.epsilon())


@keras.saving.register_keras_serializable()
class WeightedMaeMapeLoss(keras.losses.Loss):
    def __init__(self, mae_weight=1.0, mape_weight=1.0, **kwargs):
        super().__init__(**kwargs)
        self.mae_weight = mae_weight
        self.mape_weight = mape_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        abs_diff = tf.abs(y_true - y_pred)
        mae = tf.reduce_mean(abs_diff)
        mape = tf.reduce_mean((abs_diff / (tf.abs(y_true) + K.epsilon())) * 100)

        loss = (self.mae_weight * mae) + (self.mape_weight * mape)

        return loss

    def get_config(self):
        config = super().get_config()
        config.update({"mae_weight": self.mae_weight,
                       "mape_weight": self.mape_weight})
        return config
