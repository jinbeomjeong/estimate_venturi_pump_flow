import numpy as np
import tensorflow as tf
from tensorflow import keras
from utils.model import time_mixer_block
from utils.layer import InceptionBlock1D, ScalingLayer, FeatureWiseScalingLayer, gelu_approximate
from utils.layer import ChannelSelect, MultiScaleSmoothing, ChannelGate, ScaledResidual
from utils.metric import smape, WeightedMaeMapeLoss
from utils.miscellaneous import count_divisions_by_two


def get_strategy():
    """
    Returns a distribution strategy for the GPUs that are actually present.

    Creating a MirroredStrategy over a hard-coded two-GPU list at import time made
    this module unimportable on a single-GPU or CPU-only machine, so the devices
    are discovered instead. Wrap the *model construction* in the returned scope --
    wrapping only a function definition has no effect.
    """
    gpus = tf.config.list_logical_devices('GPU')

    if len(gpus) > 1:
        return tf.distribute.MirroredStrategy(devices=[gpu.name for gpu in gpus])

    return tf.distribute.get_strategy()


def build_model(input_shape=(1, 1), dropout_rate=0.2):
    """
    Builds a multi-stage Inception-based 1D Convolutional Neural Network.
    
    The model consists of four stages, where each stage's output is concatenated 
    with the original input to refine the next stage's prediction.
    
    Args:
        input_shape (tuple): Shape of the input data (timesteps, features).
        dropout_rate (float): Dropout rate for regularization.
        
    Returns:
        keras.Model: Compiled Keras model with 4 outputs.
    """
    input_layer = keras.layers.Input(shape=input_shape, name='input_layer')
    y1 = input_layer

    # Stage 1: Initial feature extraction using Inception blocks
    for i in range(3):
        y1_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=64, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y1)
        y1_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y1_1)
        y1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y1_1)

    y1 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y1)
    y1 = keras.layers.Dropout(dropout_rate)(y1)
    y1 = keras.layers.Dense(units=1, activation='linear', name='output_1')(y1)

    # Stage 2: Refinement using Stage 1 output and original input
    y2 = keras.layers.concatenate(inputs=[y1, input_layer], axis=2)

    for i in range(3):
        y2_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y2)
        y2_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y2_1)
        y2 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y2_1)

    y2 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y2)
    y2 = keras.layers.Dropout(dropout_rate)(y2)
    y2 = keras.layers.Dense(units=1, activation='linear')(y2)
    y2_add = keras.layers.add(inputs=[y2, y1], name='output_2')

    # Stage 3: Refinement using Stage 2 output and original input
    y3 = keras.layers.concatenate(inputs=[y2_add, input_layer], axis=2)

    for i in range(3):
        y3_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y3)
        y3_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y3_1)
        y3 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y3_1)

    y3 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y3)
    y3 = keras.layers.Dropout(dropout_rate)(y3)
    y3 = keras.layers.Dense(units=1, activation='linear')(y3)
    y3_add = keras.layers.add(inputs=[y3, y2_add], name='output_3')

    # Stage 4: Final refinement
    y4 = keras.layers.concatenate(inputs=[y3_add, input_layer], axis=2)

    for j in range(3):
        y4_1 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y4)
        y4_1 = keras.layers.Conv1D(filters=input_shape[1], kernel_size=3, strides=1, activation='gelu', padding='same')(y4_1)
        y4 = InceptionBlock1D(output_dim_1x1=384, hidden_dim_3x3=192, output_dim_3x3=384, hidden_dim_5x5=48, output_dim_5x5=128,
                                hidden_dim_7x7=32, output_dim_7x7=64, output_dim_max_pool=128, dropout_rate=dropout_rate)(y4_1)

    y4 = keras.layers.Conv1D(filters=input_shape[0], kernel_size=3, strides=1, activation='gelu', padding='same')(y4)
    y4 = keras.layers.Dropout(dropout_rate)(y4)
    y4 = keras.layers.Dense(units=1, activation='linear')(y4)

    y4_add = keras.layers.add([y4, y3_add], name='output_4')

    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    model = keras.models.Model(inputs=input_layer, outputs=[y1, y2_add, y3_add, y4_add])
    
    # Compile the model with multiple outputs and respective weights
    model.compile(optimizer=optimizer, loss={'output_1': 'mse', 'output_2': 'mse', 'output_3': 'mse', 'output_4': 'mse'},
                  loss_weights={'output_1': 1.0, 'output_2': 1.0, 'output_3': 1.0, 'output_4': 1.0},
                  metrics={'output_1': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_2': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_3': ['mean_absolute_error', 'mean_absolute_percentage_error', smape],
                           'output_4': ['mean_absolute_error', 'mean_absolute_percentage_error', smape]})
    return model


def build_reg_model(input_shape, pressure_scale=(0, 1), d_dims=64, dropout_rate=0.2, learning_rate=0.001):
    """
    Builds a regression model with residual dilated 1D convolutions.
    
    Args:
        input_shape (tuple): Shape of the input data.
        pressure_scale (tuple): Scale and offset for rescaling the first two features.
        d_dims (int): Number of filters in Conv1D layers.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Initial learning rate for Adam optimizer.
        
    Returns:
        keras.Model: Compiled Keras regression model.
    """
    input_layer = keras.layers.Input(shape=input_shape)
    
    # Preprocessing: Rescaling pressure features and GroupNormalization for other features
    x1 = keras.layers.Rescaling(scale=pressure_scale[0], offset=pressure_scale[1])(input_layer[:, :, 0:2])
    x2 = keras.layers.GroupNormalization(groups=1)(input_layer[:, :, 2:])
    x = keras.layers.concatenate([x1, x2], axis=2)

    x_res = keras.layers.Dense(units=d_dims, activation=gelu_approximate)(x)

    # Residual blocks with dilated convolutions
    for i in range(count_divisions_by_two(input_shape[0])+1):
        dilation_rate = 2 ** i
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation=gelu_approximate, padding='causal',
                                dilation_rate=dilation_rate)(x_res)
        x = keras.layers.Dropout(dropout_rate)(x)
        x = keras.layers.Conv1D(filters=d_dims, kernel_size=3, activation=gelu_approximate, padding='causal',
                                dilation_rate=dilation_rate)(x)

        # Residual connection with BatchNormalization
        x_res = keras.layers.BatchNormalization()(x + x_res)
        x_res = keras.layers.Activation(gelu_approximate)(x_res)

    # Final layers for regression output
    y = keras.layers.Flatten()(x_res)
    y = keras.layers.GroupNormalization(groups=1)(y)
    y = keras.layers.Dropout(dropout_rate)(y)

    y = FeatureWiseScalingLayer()(y)
    y = keras.layers.Dropout(dropout_rate)(y)
    y = keras.layers.Dense(units=1, activation='linear')(y)

    model = keras.models.Model(inputs=input_layer, outputs=y)

    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

    # Log-Cosh loss is often smoother than MSE for regression
    model.compile(optimizer=optimizer, loss=keras.losses.logcosh,
                  metrics=['mean_absolute_error','mean_absolute_percentage_error'])

    return model


def build_flow_model_v2(input_shape=(20, 3), feature_stats=((0.0, 0.0), (1.0, 1.0)), target_stats=(0.0, 1.0),
                        feature_channels=(0, 1), width=32, dropout_rate=0.1, weight_decay=1e-4,
                        gate_l2=1e-2, gate_init=(0.3, 1.0), spans=(1, 3, 5, 10, 20), learning_rate=3e-3,
                        loss=None):
    """
    Builds the venturi flow estimator: a linear trunk with a small gated nonlinear correction.

    Why this shape rather than a deep dilated CNN. Measured on the logged data:

      * a plain linear fit on the 20-step averaged pressures already reaches the same
        validation accuracy as the 500k-parameter dilated CNN it replaces, so the
        capacity of that network was going into memorising the training session;
      * the window helps by averaging out sensor noise, not by carrying dynamics
        (last sample 4.75% MAPE vs 20-step mean 4.34%), so the averages are handed
        to the network directly;
      * within one session the pressure-to-flow relation is close to linear
        (R2 0.99 on the validation session), while the *bias* moves from session to
        session, which is variance a large model makes worse, not better.

    So the trunk stays linear and generalises, and everything nonlinear has to come
    through ScaledResidual, whose alpha starts at zero.

    The input keeps its (timesteps, 3) shape so the logging and inference buffers do
    not have to change, but only `feature_channels` reaches the network -- pump speed
    is excluded structurally, not by hoping a normalisation layer discards it.

    Args:
        input_shape (tuple): Shape of the input window, e.g. (20, 3).
        feature_stats (tuple): (mean, std) per selected channel, from the training set.
        target_stats (tuple): (mean, std) of the training flow rate, folded into the
            output layer so the model emits LPM while learning a standardised target.
        feature_channels (tuple): Channel indices the model is allowed to see.
        width (int): Hidden units of the nonlinear correction branch.
        dropout_rate (float): Dropout rate of the correction branch.
        weight_decay (float): L2 penalty on the dense kernels.
        gate_l2 (float): L2 penalty on the per-channel input gate.
        gate_init (tuple): Initial gate value per selected channel.
        spans (tuple): Trailing window lengths averaged by MultiScaleSmoothing.
        learning_rate (float): Initial learning rate for Adam.
        loss: Keras loss. Defaults to log-cosh, which behaves like MAE on LPM-scale errors.

    Returns:
        keras.Model: Compiled Keras model emitting one flow rate in LPM.
    """
    feature_mean = np.asarray(feature_stats[0], dtype=np.float32)
    feature_std = np.asarray(feature_stats[1], dtype=np.float32)
    target_mean, target_std = float(target_stats[0]), float(target_stats[1])

    input_layer = keras.layers.Input(shape=input_shape, name='input_layer')

    # Keep only the channels the model is allowed to use.
    x = ChannelSelect(channels=feature_channels, name='select_channels')(input_layer)

    # Fixed affine scaling from training statistics. Never per sample: the absolute
    # pressure level is what the flow rate is read from, so normalising each window
    # against itself would throw the signal away.
    x = keras.layers.Rescaling(scale=1.0 / feature_std, offset=-feature_mean / feature_std,
                               name='input_scaling')(x)
    # 게이트 초깃값은 실제로 선택된 채널 수와 길이가 같아야 합니다. 길이가 다르면
    # 브로드캐스팅으로 채널 수가 바뀌므로 ChannelGate가 바로 막아 세웁니다.
    if len(gate_init) != len(feature_channels):
        gate_init = (1.0,) * len(feature_channels)

    x = ChannelGate(l2=gate_l2, init=gate_init, name='channel_gate')(x)

    features = MultiScaleSmoothing(spans=spans, name='multi_scale')(x)

    # Linear trunk: the part of the mapping that holds across sessions.
    trunk = keras.layers.Dense(units=1, activation='linear', name='linear_trunk',
                               kernel_regularizer=keras.regularizers.L2(weight_decay))(features)

    # Nonlinear correction, kept small and gated by a scalar that starts at zero.
    hidden = keras.layers.Dense(units=width, activation=gelu_approximate,
                                kernel_regularizer=keras.regularizers.L2(weight_decay))(features)
    hidden = keras.layers.Dropout(dropout_rate)(hidden)
    hidden = keras.layers.Dense(units=width, activation=gelu_approximate,
                                kernel_regularizer=keras.regularizers.L2(weight_decay))(hidden)
    hidden = keras.layers.Dropout(dropout_rate)(hidden)
    correction = keras.layers.Dense(units=1, activation='linear', name='correction',
                                    kernel_regularizer=keras.regularizers.L2(weight_decay))(hidden)

    standardised = ScaledResidual(init=0.0, name='residual_mix')([trunk, correction])

    # Fold the target statistics into the graph so the deployed model emits LPM.
    output_layer = keras.layers.Rescaling(scale=target_std, offset=target_mean, name='flow_rate_lpm')(standardised)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss=keras.losses.logcosh if loss is None else loss,
                  metrics=['mean_absolute_error', 'mean_absolute_percentage_error'])

    return model
