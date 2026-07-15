import keras
from utils.miscellaneous import count_divisions_by_two


def build_reg_model(input_shape, d_dims=64, dropout_rate=0.2, learning_rate=0.001):
    input_layer = keras.layers.Input(shape=input_shape)
    # 모든 채널을 채널별 통계로 정규화한다. 단일 채널에 LayerNorm(axis=-1)을 적용하면
    # 값이 항상 상수(beta)로 붕괴되어 해당 feature 신호가 소거되므로 BatchNormalization으로 통일한다.
    x = keras.layers.BatchNormalization()(input_layer)

    x_res = keras.layers.Dense(units=d_dims, activation='gelu')(x)

    for i in range(count_divisions_by_two(input_shape[0])+1):
        dilation_rate = 2 ** i
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                dilation_rate=dilation_rate)(x_res)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation='gelu', padding='causal',
                                dilation_rate=dilation_rate)(x)

        # 잔차 합 이후 활성화를 제거해 항등(identity) 지름길을 보존한다.
        x_res = keras.layers.BatchNormalization()(x + x_res)

    # 회귀 헤드: 시간축 평균풀링 → 선형 readout.
    # (헤드 A/B 비교 결과 pre-readout LayerNorm을 제거한 이 구조가 저유량 상대오차·MAE에서 최적이었다.)
    y = keras.layers.GlobalAveragePooling1D()(x_res)
    y = keras.layers.Dropout(dropout_rate)(y)
    y = keras.layers.Dense(units=1, activation='linear')(y)

    model = keras.models.Model(inputs=input_layer, outputs=y)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer, loss=keras.losses.LogCosh(),
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
