import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def make_model_def():
    """
    Adopted from:
        https://github.com/miquelmarti/tiny-imagenet-classifier/blob/master/tiny_imagenet_classifier.py
    """
    model_inp = tf.keras.layers.Input(shape=(64, 64, 3))

    # conv-spatial batch norm - relu #1
    x = tf.keras.layers.ZeroPadding2D((2, 2))(model_inp)
    x = tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                   kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-7, l2=1e-7))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #2
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #3
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # conv-spatial batch norm - relu #4
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(128, (3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #5
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #6
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(256, (3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # conv-spatial batch norm - relu #7
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #8
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(512, (3, 3), strides=(1, 1))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #9
    x = tf.keras.layers.ZeroPadding2D((1, 1))(x)
    x = tf.keras.layers.Conv2D(1024, (3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.25)(x)

    # Affine-spatial batch norm -relu #10
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512,
                              kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))(x)
    x = tf.keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    model_out = tf.keras.layers.Dense(
        FLAGS.N_CLASSES, activation='softmax',
        kernel_regularizer=tf.keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)
    )(x)
    model = tf.keras.models.Model(model_inp, model_out)

    return model
