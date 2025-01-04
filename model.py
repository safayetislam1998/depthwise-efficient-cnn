import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers

def normalization(inputs):
    return tf.keras.layers.BatchNormalization()(inputs)

def create_model(image_size=32, projection_dim=128, depth=8, patch_size=2, num_classes=10, drop_rate=0.1):
    """
    Args:
        image_size: Input image size
        projection_dim: Dimension of the projection
        depth: Number of blocks
        patch_size: Size of the patches
        num_classes: Number of output classes
        drop_rate: Dropout rate
        
    Returns:
        tf.keras.Model: Compiled model
    """
    inputs = layers.Input((image_size, image_size, 3))

    # Initial convolution stem
    conv = layers.Conv2D(filters=projection_dim, kernel_size=patch_size, strides=patch_size)(inputs)
    conv = layers.Activation('gelu')(conv)
    conv = normalization(conv)
    
    # Main architecture blocks
    for i in range(depth):
        root = conv
        
        # First 1x1 conv for channel reduction
        nn = layers.Conv2D(filters=projection_dim//2, kernel_size=1)(conv)
        nn = layers.Activation('gelu')(nn)
        nn = normalization(nn)
        
        # Second 1x1 conv for further channel reduction
        nn = layers.Conv2D(filters=projection_dim//4, kernel_size=1)(nn)
        nn = layers.Activation('gelu')(nn)
        nn = normalization(nn)
        
        # Depthwise convolution
        depth = nn
        nn = layers.DepthwiseConv2D(kernel_size=5, padding='same', name=f'depth_conv{i}')(nn)
        nn = layers.Activation('gelu')(nn)
        nn = normalization(nn)
        nn = layers.Add()([nn, depth])
        
        # Pointwise convolution
        point = nn
        nn = layers.Conv2D(filters=projection_dim, kernel_size=1, name=f'point_conv{i}')(nn)
        nn = layers.Activation('gelu')(nn)
        nn = layers.Add()([root, nn])
        conv = normalization(nn)

    representation = conv
    flatten = tfa.layers.AdaptiveAveragePooling2D((1,1))(representation)
    flatten = layers.Flatten()(flatten)
    logits = layers.Dense(num_classes, activation='softmax')(flatten)
    
    return tf.keras.Model(inputs=inputs, outputs=logits)

# Model variants
def get_small_model():
    return create_model(projection_dim=128, depth=8, patch_size=2)

def get_medium_model():
    return create_model(projection_dim=256, depth=8, patch_size=2)

def get_large_model():
    return create_model(projection_dim=256, depth=16, patch_size=1)