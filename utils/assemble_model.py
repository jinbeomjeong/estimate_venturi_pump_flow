import tensorflow as tf
from tensorflow import keras
from utils.model import time_mixer_block
from utils.layer import InceptionBlock1D, ScalingLayer, FeatureWiseScalingLayer, gelu_approximate
from utils.metric import smape, WeightedMaeMapeLoss
from utils.miscellaneous import count_divisions_by_two


# Setup distributed training strategy using available GPUs
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])


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


with strategy.scope():
    def build_reg_model(input_shape, pressure_scale=(0, 1), d_dims=64, dropout_rate=0.2, learning_rate=0.001):
        """
        Builds a regression model with residual dilated 1D convolutions within a distribution strategy.
        
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
