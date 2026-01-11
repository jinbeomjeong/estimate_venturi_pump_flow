import keras, tf2onnx, logging
import tensorflow as tf

from utils.layer import FeatureWiseScalingLayer, gelu_approximate


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

seq_len = 20
model_path = f'../models/model_{seq_len}.keras'

model = keras.models.load_model(filepath=model_path, custom_objects={'FeatureWiseScalingLayer':FeatureWiseScalingLayer,
                                                                     'gelu_approximate': gelu_approximate})

logging.info(f'Model loaded from {model_path}')

spec = (tf.TensorSpec(model.inputs[0].shape, tf.float32, name='input'),)
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec)
logging.info('converted ONNX model')

with open(f'../models/model_{seq_len}.onnx', "wb") as f:
    f.write(onnx_model.SerializeToString())

logging.info('saved ONNX model')
