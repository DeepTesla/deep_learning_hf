import keras

def CustomNetwork1():
    kernel_size = (6,8)
    filters = 64

    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(120, 160, 3)))    # RGB image goes in

    # ENCODER
    # --------------------
    for _ in range(3):
        for _ in range(2):
            model.add(keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            padding='same',
                                            use_bias=True,
                                            bias_initializer=keras.initializers.constant(value=0.0)))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.ReLU())

        model.add(keras.layers.MaxPool2D(pool_size=2, strides=2))
        filters *= 2
    # --------------------

    # DECODER
    # --------------------
    for _ in range(3):
        filters = round(filters/2)
        model.add(keras.layers.UpSampling2D(size=2,
                                            interpolation='nearest'))
        for _ in range(2):
            model.add(keras.layers.Conv2D(filters=filters,
                                            kernel_size=kernel_size,
                                            strides=1,
                                            padding='same',
                                            use_bias=True,
                                            bias_initializer=keras.initializers.constant(value=0.0)))
            model.add(keras.layers.BatchNormalization())       
            model.add(keras.layers.ReLU())
    # --------------------

    model.add(keras.layers.Dense(units=1,
                                    activation='sigmoid',
                                    use_bias=True
    ))
    model.add(keras.layers.Reshape((120,160)))

    return model
