import tensorflow as tf

def conv_block(inputs, n_filters, kernel_size=3, strides=(1, 1), dilation_rate=(1, 1), batchnorm=True, dropout_rate=0.0, activation='relu'):
    """
    Convolutional block consisting of two convolutional layers with optional batch normalization and dropout.
    
    Parameters:
    - inputs: Input tensor to the block.
    - n_filters: Number of filters for the convolutional layers.
    - kernel_size: Size of the convolutional kernel.
    - strides: Strides for the convolutional layers.
    - dilation_rate: Dilation rate for the convolutional layers.
    - batchnorm: Boolean to include batch normalization.
    - dropout_rate: Dropout rate (if dropout is applied).
    - activation: Activation function to apply after each convolution.
    
    Returns:
    - x: Output tensor after applying the convolutions, batch normalization, and activation functions.
    """
    x = inputs
    for i in range(2):
        x = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=strides,
                                   dilation_rate=dilation_rate, padding='same', 
                                   kernel_initializer=tf.keras.initializers.HeNormal())(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation(activation)(x)
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate)(x)
    return x


def attention_block(x, g, inter_channel, mode='additive'):
    """
    Attention block that computes an attention map and applies it to the input feature map.
    
    Parameters:
    - x: Input feature map.
    - g: Gated feature map.
    - inter_channel: Intermediate number of filters.
    - mode: Mode of attention mechanism ('additive' or 'multiplicative').
    
    Returns:
    - rate: Output after applying attention to the input feature map.
    """
    theta_x = conv_block(x, inter_channel, kernel_size=1, batchnorm=False)
    phi_g = conv_block(g, inter_channel, kernel_size=1, batchnorm=False)
    
    if mode == 'additive':
        f = tf.keras.layers.Add()([theta_x, phi_g])
    elif mode == 'multiplicative':
        f = tf.keras.layers.Multiply()([theta_x, phi_g])
    
    f = tf.keras.layers.ReLU()(f)
    
    psi_f = tf.keras.layers.Conv2D(1, kernel_size=1, activation='sigmoid',
                                   kernel_initializer=tf.keras.initializers.HeNormal(), padding='same')(f)
    
    rate = tf.keras.layers.Multiply()([psi_f, x])
    return rate


def triplet_attention(inputs, inter_channel, aggregation_mode='concatenate', attention_mode='additive'):
    """
    Triplet Attention block that aggregates attention across three pathways.
    
    Parameters:
    - inputs: Input tensor to the block.
    - inter_channel: Intermediate number of filters for the convolutional layers in attention pathways.
    - aggregation_mode: Mode of aggregating attention outputs ('concatenate' or 'sum').
    - attention_mode: Mode of attention mechanism to be used in each pathway ('additive' or 'multiplicative').
    
    Returns:
    - output: Aggregated output after applying triplet attention.
    """
    g1 = conv_block(inputs, inter_channel, kernel_size=1, batchnorm=False)
    g2 = conv_block(inputs, inter_channel, kernel_size=1, batchnorm=False)
    g3 = conv_block(inputs, inter_channel, kernel_size=1, batchnorm=False)
    
    att1 = attention_block(inputs, g1, inter_channel, mode=attention_mode)
    att2 = attention_block(inputs, g2, inter_channel, mode=attention_mode)
    att3 = attention_block(inputs, g3, inter_channel, mode=attention_mode)
    
    if aggregation_mode == 'concatenate':
        output = tf.keras.layers.Concatenate(axis=-1)([att1, att2, att3])
    elif aggregation_mode == 'sum':
        output = tf.keras.layers.Add()([att1, att2, att3])
    
    return output
