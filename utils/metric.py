import keras
from keras import ops


def smape(y_true, y_pred):
    """
    Keras용 SMAPE(Symmetric Mean Absolute Percentage Error) 메트릭 함수

    수식: 100/n * Σ(|y_pred - y_true| / ((|y_true| + |y_pred|)/2))
    """
    # numpy 배열로 호출되어도 동작하도록 백엔드 텐서로 변환한다.
    y_true = ops.cast(ops.convert_to_tensor(y_true), 'float32')
    y_pred = ops.cast(ops.convert_to_tensor(y_pred), 'float32')

    # 0으로 나누는 것을 방지하기 위해 작은 epsilon 값을 더해줍니다.

    # 분자: 예측값과 실제값의 차이의 절댓값
    numerator = ops.abs(y_pred - y_true)

    # 분모: 실제값과 예측값의 절댓값의 합을 2로 나눔
    denominator = (ops.abs(y_true) + ops.abs(y_pred)) / 2 + keras.backend.epsilon()

    # SMAPE 계산
    # 각 데이터 포인트에 대한 백분율 오류를 계산하고 평균을 냅니다.
    percent_error = (numerator / denominator) * 100

    return ops.mean(percent_error)


def r_squared(y_true, y_pred):
    """
    R-squared 계산 함수
    """
    # numpy 배열로 호출되어도 동작하도록 백엔드 텐서로 변환한다.
    y_true = ops.cast(ops.convert_to_tensor(y_true), 'float32')
    y_pred = ops.cast(ops.convert_to_tensor(y_pred), 'float32')

    SS_res = ops.sum(ops.square(y_true - y_pred))
    SS_tot = ops.sum(ops.square(y_true - ops.mean(y_true)))
    return 1 - SS_res / (SS_tot + keras.backend.epsilon())
