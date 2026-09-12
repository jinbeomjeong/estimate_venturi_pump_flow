import os, keras, tf2onnx, logging
import tensorflow as tf

from utils.layer import FeatureWiseScalingLayer, gelu_approximate
from utils.layer import ChannelSelect, MultiScaleSmoothing, ChannelGate, ScaledResidual


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seq_len = 20
model_name = f'model_{seq_len}_v2'
model_path = os.path.join('models', f'{model_name}.keras')

# 커스텀 레이어는 이름으로 찾아지지 않으므로 전부 넘겨줘야 합니다.
custom_objects = {'FeatureWiseScalingLayer': FeatureWiseScalingLayer,
                  'gelu_approximate': gelu_approximate,
                  'ChannelSelect': ChannelSelect,
                  'MultiScaleSmoothing': MultiScaleSmoothing,
                  'ChannelGate': ChannelGate,
                  'ScaledResidual': ScaledResidual}

model = keras.models.load_model(filepath=model_path, custom_objects=custom_objects)

logging.info(f'Model loaded from {model_path}')

# 배치 축은 열어두고 (시간, 채널)만 고정합니다. 추론 쪽 입력 이름은 'input'입니다.
input_shape = [None] + list(model.inputs[0].shape[1:])
spec = (tf.TensorSpec(input_shape, tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
logging.info('converted ONNX model')

onnx_path = os.path.join('models', f'{model_name}.onnx')

with open(onnx_path, 'wb') as f:
    f.write(onnx_model.SerializeToString())

logging.info(f'saved ONNX model to {onnx_path}')
