def conv_block(inputs, n_filters, kernel_size=3, batchnorm=True):
    x = inputs
    for _ in range(2):
        x = tf.keras.layers.Conv2D(n_filters, kernel_size, activation='relu', kernel_initializer='he_normal', padding='same')(x)
        if batchnorm:
            x = tf.keras.layers.BatchNormalization()(x)
    return x


def attention_block(x, g, inter_channel):
    theta_x = conv_block(x, inter_channel, kernel_size=1, batchnorm=False)
    phi_g = conv_block(g, inter_channel, kernel_size=1, batchnorm=False)
    f = tf.keras.layers.ReLU()(theta_x + phi_g)
    psi_f = tf.keras.layers.Conv2D(1, 1, activation='sigmoid', kernel_initializer='he_normal', padding='same')(f)
    rate = psi_f * x
    return rate


# Define the triplet_attention function
def triplet_attention(x, inter_channel):
    g1 = conv_block(x, inter_channel, kernel_size=1, batchnorm=False)
    g2 = conv_block(x, inter_channel, kernel_size=1, batchnorm=False)
    g3 = conv_block(x, inter_channel, kernel_size=1, batchnorm=False)
    att1 = attention_block(x, g1, inter_channel)
    att2 = attention_block(x, g2, inter_channel)
    att3 = attention_block(x, g3, inter_channel)
    y = tf.concat([att1, att2, att3], axis=-1)
    return y
