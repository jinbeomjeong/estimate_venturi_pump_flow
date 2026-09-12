"""
벤츄리 펌프 유량 추정 모델(v2) 학습 스크립트.

노트북 대신 스크립트로 둔 이유는, 같은 명령으로 같은 숫자가 다시 나오게 하기
위해서입니다. 학습 세션과 평가 세션이 다르기 때문에 세 갈래로 나눠 씁니다.

  학습 (data/train_data)          : 가중치를 맞추는 데 사용
  검증 (data/test_data/2025-11-17): 조기 종료와 학습률 조정에 사용 -> 낙관적인 값
  테스트(data/test_data/2025-11-14): 아무 데도 쓰지 않음 -> 이 숫자만 믿으면 됩니다

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


SEQ_LEN = 20
PERIOD = 0.5
TRAIN_DIR = os.path.join('data', 'train_data')
VAL_FILES = [os.path.join('data', 'test_data', name) for name in
             ['data20251117-142801.csv', 'data20251117-143219.csv', 'data20251117-143505.csv',
              'data20251117-143657.csv', 'data20251117-143831.csv']]
TEST_FILES = [os.path.join('data', 'test_data', 'data20251114-170633.csv')]
FEATURE_CHANNELS = (1,)     # 토출 압력만. 회전수와 흡입 압력은 쓰지 않습니다(MODEL.md 참고).


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
    return (f'{name:22s} R2 {score["r2"]:6.4f}  MAE {score["mae"]:7.1f} LPM  '
            f'MAPE {score["mape"]:5.2f}%  |err|p95 {score["p95"]:7.1f}  bias {score["bias"]:+7.1f}')


def train_one(train_data, val_data, stats, seed, epochs, batch_size, verbose):
    (train_feature, train_target), (val_feature, val_target) = train_data, val_data
    keras.utils.set_random_seed(seed)

    with get_strategy().scope():
        model = build_flow_model_v2(input_shape=train_feature.shape[1:],
                                    feature_stats=stats['feature'], target_stats=stats['target'],
                                    feature_channels=FEATURE_CHANNELS)

    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=60, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=25, min_lr=1e-5)]

    model.fit(x=train_feature, y=train_target, validation_data=(val_feature, val_target),
              epochs=epochs, batch_size=batch_size, verbose=verbose, callbacks=callbacks)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3],
                        help='앙상블에 쓸 시드. 데이터가 3천여 개뿐이라 시드별 편차가 큽니다.')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--no-save', action='store_true')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    (train_feature, train_target), (val_feature, val_target) = load_flow_dataset(
        TRAIN_DIR, VAL_FILES, seq_len=SEQ_LEN, period=PERIOD)
    _, (test_feature, test_target) = load_flow_dataset(
        TRAIN_DIR, TEST_FILES, seq_len=SEQ_LEN, period=PERIOD)

    print(f'train {train_feature.shape}  val {val_feature.shape}  test {test_feature.shape}')

    used = train_feature[:, :, list(FEATURE_CHANNELS)].reshape(-1, len(FEATURE_CHANNELS))
    stats = {'feature': (used.mean(axis=0), used.std(axis=0)),
             'target': (float(train_target.mean()), float(train_target.std()))}
    print('feature mean/std', stats['feature'], 'target mean/std', stats['target'])

    predictions = {'train': [], 'val': [], 'test': []}
    model = None

    for seed in args.seeds:
        model = train_one((train_feature, train_target), (val_feature, val_target), stats,
                          seed, args.epochs, args.batch_size, args.verbose)

        for key, feature in (('train', train_feature), ('val', val_feature), ('test', test_feature)):
            predictions[key].append(model.predict(feature, verbose=0, batch_size=512).ravel())

        alpha = float(model.get_layer('residual_mix').alpha.numpy()[0])
        gate = model.get_layer('channel_gate').gate.numpy()
        # alpha 는 비선형 보정이 실제로 얼마나 개입하는지, gate 는 각 입력 채널을
        # 얼마나 신뢰하고 있는지를 한 숫자로 보여줍니다.
        print(f'  seed {seed}: residual alpha {alpha:+.4f}  channel gate {np.round(gate, 4)}')

    print(f'\n--- {len(args.seeds)} seed ensemble ({model.count_params():,} parameters) ---')
    print(report('train', evaluate(train_target, np.mean(predictions['train'], axis=0))))
    print(report('val (used to stop)', evaluate(val_target, np.mean(predictions['val'], axis=0))))
    print(report('test (held out)', evaluate(test_target, np.mean(predictions['test'], axis=0))))

    if not args.no_save:
        os.makedirs('models', exist_ok=True)
        path = os.path.join('models', f'model_{SEQ_LEN}_v2.keras')
        model.save(path)
        print(f'\nsaved {path} (last seed; run utils/converte_onnx.py to export it)')


if __name__ == '__main__':
    main()
