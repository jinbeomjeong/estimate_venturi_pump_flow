import keras
from keras import ops


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
