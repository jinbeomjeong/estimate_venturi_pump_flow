from tensorflow.keras import backend as K


def smape(y_true, y_pred):
    """
    Keras용 SMAPE(Symmetric Mean Absolute Percentage Error) 메트릭 함수

    수식: 100/n * Σ(|y_pred - y_true| / ((|y_true| + |y_pred|)/2))
    """
    # 0으로 나누는 것을 방지하기 위해 작은 epsilon 값을 더해줍니다.
    epsilon = 1e-8

    # 분자: 예측값과 실제값의 차이의 절댓값
    numerator = K.abs(y_pred - y_true)

    # 분모: 실제값과 예측값의 절댓값의 합을 2로 나눔
    denominator = (K.abs(y_true) + K.abs(y_pred)) / 2 + epsilon

    # SMAPE 계산
    # 각 데이터 포인트에 대한 백분율 오류를 계산하고 평균을 냅니다.
    percent_error = (numerator / denominator) * 100

    return K.mean(percent_error)