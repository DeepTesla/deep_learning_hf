import tensorflow as tf

filters = 64
kernel_size = (6, 8)

model12 = tf.keras.models.Sequential()

# ENCODER

model12.add(tf.keras.layers.InputLayer(input_shape=(60, 80, 3)))
# --------------------
model12.add(tf.keras.layers.Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.BatchNormalization())
model12.add(tf.keras.layers.ReLU())

model12.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# --------------------
model12.add(tf.keras.layers.Conv2D(filters=filters * 2,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.Conv2D(filters=filters * 2,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.BatchNormalization())
model12.add(tf.keras.layers.ReLU())

model12.add(tf.keras.layers.MaxPool2D(pool_size=2, strides=2))
# --------------------a

# DECODER

# --------------------
model12.add(tf.keras.layers.UpSampling2D(size=2,
                                       interpolation='nearest'))
model12.add(tf.keras.layers.Conv2D(filters=filters * 2,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.Conv2D(filters=filters * 2,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.BatchNormalization())
model12.add(tf.keras.layers.ReLU())
# --------------------
model12.add(tf.keras.layers.UpSampling2D(size=2,
                                       interpolation='nearest'))
model12.add(tf.keras.layers.Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 strides=1,
                                 padding='same',
                                 use_bias=True,
                                 bias_initializer=tf.keras.initializers.constant(value=0.0)))
model12.add(tf.keras.layers.BatchNormalization())
model12.add(tf.keras.layers.ReLU())
model12.add(tf.keras.layers.Dense(units=1,
                                activation='sigmoid',
                                use_bias=True))
model12.add(tf.keras.layers.Reshape((60, 80)))
