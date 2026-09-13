"""
벤츄리 펌프 유량 추정 모델(v2) 학습 스크립트.

노트북 대신 스크립트로 둔 이유는, 같은 명령으로 같은 숫자가 다시 나오게 하기
위해서입니다.

평가 방식:
  학습   data/train_data           뒤쪽 15%는 조기 종료에만 씁니다
  평가   data/test_data 의 6개 세션   학습에 전혀 관여하지 않습니다

조기 종료를 로그 세션이 아니라 학습 데이터 안에서 끊는 이유는, 세션으로 끊으면
그 세션에 맞춰진 모델이 나오기 때문입니다. 이전 구조가 검증 세션에서 MAPE
1.98%를 내고 처음 보는 세션에서 9.49%로 무너진 것이 정확히 그 경로였습니다.
이렇게 두면 여섯 세션이 전부 정직한 평가 세트가 됩니다.

사용법:
    python train_flow_model.py                # 학습 후 models/model_20_v2.keras 저장
    python train_flow_model.py --no-save      # 숫자만 확인
"""
import os
import argparse
import numpy as np

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

from tensorflow import keras

from utils.dataset import load_flow_dataset
from utils.assemble_model import build_flow_model_v2, get_strategy
from utils.layer import DifferentialPressure


SEQ_LEN = 20
PERIOD = 0.5
STOP_SPLIT = 0.15           # 조기 종료에 쓸 학습 데이터 뒤쪽 비율
TRAIN_DIR = os.path.join('data', 'train_data')
TEST_DIR = os.path.join('data', 'test_data')
EVAL_SESSIONS = ['data20251114-170633.csv', 'data20251117-142801.csv', 'data20251117-143219.csv',
                 'data20251117-143505.csv', 'data20251117-143657.csv', 'data20251117-143831.csv']
FEATURE_CHANNELS = (0, 1)   # 흡입 압력, 토출 압력. 모델 안에서 차압으로 바뀝니다(MODEL.md 참고).


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    error = y_pred - y_true
    ss_res = float(np.sum(error ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))

    return {'r2': 1.0 - (ss_res / ss_tot),
            'mae': float(np.mean(np.abs(error))),
            'mape': float(np.mean(np.abs(error) / np.abs(y_true)) * 100),
            'p95': float(np.percentile(np.abs(error), 95)),
            'bias': float(np.mean(error))}


def report(name: str, score: dict) -> str:
    return (f'{name:24s} R2 {score["r2"]:7.4f}  MAE {score["mae"]:7.1f} LPM  '
            f'MAPE {score["mape"]:5.2f}%  |err|p95 {score["p95"]:7.1f}  bias {score["bias"]:+7.1f}')


def load_sessions() -> tuple:
    """학습 창과, 세션별로 따로 담은 평가 창을 돌려줍니다."""
    (feature, target), _ = load_flow_dataset(TRAIN_DIR, [], seq_len=SEQ_LEN, period=PERIOD)
    sessions = {}

    for file_name in EVAL_SESSIONS:
        _, pair = load_flow_dataset(TRAIN_DIR, [os.path.join(TEST_DIR, file_name)],
                                    seq_len=SEQ_LEN, period=PERIOD)
        # 11-17 세션은 날짜가 같으므로 시각까지 키에 넣어야 서로 덮어쓰지 않습니다.
        sessions[os.path.splitext(file_name)[0][4:]] = pair

    return (feature, target), sessions


def train_one(fit_data, stop_data, stats, seed, epochs, batch_size, verbose):
    (fit_feature, fit_target), (stop_feature, stop_target) = fit_data, stop_data
    keras.utils.set_random_seed(seed)

    with get_strategy().scope():
        model = build_flow_model_v2(input_shape=fit_feature.shape[1:],
                                    feature_stats=stats['feature'], target_stats=stats['target'],
                                    feature_channels=FEATURE_CHANNELS)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-5)]

    model.fit(x=fit_feature, y=fit_target, validation_data=(stop_feature, stop_target),
              epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='앙상블에 쓸 시드. 데이터가 3천여 개뿐이라 시드별 편차가 있습니다.')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    (train_feature, train_target), sessions = load_sessions()
    cut = int(len(train_feature) * (1.0 - STOP_SPLIT))
    fit_data = (train_feature[:cut], train_target[:cut])
    stop_data = (train_feature[cut:], train_target[cut:])

    print(f'fit {fit_data[0].shape}  stop {stop_data[0].shape}  | '
          + '  '.join(f'{name} {len(target)}' for name, (_, target) in sessions.items()))

    # 모델이 실제로 보게 될 값으로 통계를 냅니다. 같은 변환을 여기서 다시 쓰지 않고
    # 레이어를 직접 불러 쓰므로, 둘이 어긋날 일이 없습니다.
    used = fit_data[0][:, :, list(FEATURE_CHANNELS)]
    used = np.asarray(DifferentialPressure()(used)).reshape(-1, len(FEATURE_CHANNELS))
    stats = {'feature': (used.mean(axis=0), used.std(axis=0)),
             'target': (float(fit_data[1].mean()), float(fit_data[1].std()))}
    print('feature mean/std (토출압, 차압)', stats['feature'], 'target mean/std', stats['target'])

    accumulated = {name: 0.0 for name in sessions}
    model = None

    for seed in args.seeds:
        model = train_one(fit_data, stop_data, stats, seed, args.epochs, args.batch_size, args.verbose)

        for name, (feature, _) in sessions.items():
            accumulated[name] = accumulated[name] + model.predict(feature, verbose=0, batch_size=512).ravel()

        alpha = float(model.get_layer('residual_mix').alpha.numpy()[0])
        gate = model.get_layer('channel_gate').gate.numpy()
        # alpha 는 비선형 보정이 실제로 얼마나 개입하는지, gate 는 각 입력 채널을
        # 얼마나 신뢰하고 있는지를 한 숫자로 보여줍니다.
        print(f'  seed {seed}: residual alpha {alpha:+.4f}  channel gate (토출압, 차압) {np.round(gate, 4)}')

    print(f'\n--- {len(args.seeds)} seed ensemble ({model.count_params():,} parameters) ---')
    mape_list = []

    for name, (_, target) in sessions.items():
        score = evaluate(target, accumulated[name] / len(args.seeds))
        mape_list.append(score['mape'])
        print(report(name, score))

    print(f'\n세션 평균 MAPE {np.mean(mape_list):5.2f}%    최악 세션 {max(mape_list):5.2f}%')

    if not args.no_save:
        os.makedirs('models', exist_ok=True)
        path = os.path.join('models', f'model_{SEQ_LEN}_v2.keras')
        model.save(path)
        print(f'\nsaved {path} (last seed; run utils/converte_onnx.py to export it)')


if __name__ == '__main__':
    main()
