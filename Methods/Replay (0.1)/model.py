from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, concatenate, Dropout, Conv2DTranspose, Softmax




def unet_vanilla(input_size=(512, 512, 1), base=2, uncertainty=False, trainable=True):

    b = base
    inputs = Input(input_size)

    c1 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(inputs)
    c1 = Dropout(0.1)(c1, training=uncertainty)
    c1 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(c1)
    p1 = MaxPooling2D((2, 2), padding='same')(c1)

    c2 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(p1)
    c2 = Dropout(0.1)(c2, training=uncertainty)
    c2 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(c2)
    p2 = MaxPooling2D((2, 2), padding='same')(c2)

    c3 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(p2) # 9
    c3 = Dropout(0.2)(c3, training=uncertainty)
    c3 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(c3) # 11
    p3 = MaxPooling2D((2, 2), padding='same')(c3)

    c4 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(p3) # 13
    c4 = Dropout(0.2)(c4, training=uncertainty)
    c4 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable= trainable)(c4) # 15
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(2 ** (b + 4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(p4) # 17
    c5 = Dropout(0.3)(c5, training=uncertainty)
    c5 = Conv2D(2 ** (b + 4), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(c5) # 19

    u6 = Conv2DTranspose(2 ** (b + 3), (2, 2), strides=(2, 2), padding='same', trainable=trainable)(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(u6) # 22
    c6 = Dropout(0.2)(c6, training=uncertainty)
    c6 = Conv2D(2 ** (b + 3), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(c6) # 24

    u7 = Conv2DTranspose(2 ** (b + 2), (2, 2), strides=(2, 2), padding='same', trainable=trainable)(c6)                          # 25
    u7 = concatenate([u7, c3])
    c7 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(u7) # 27
    c7 = Dropout(0.2)(c7, training=uncertainty)
    c7 = Conv2D(2 ** (b + 2), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(c7) # 29

    u8 = Conv2DTranspose(2 ** (b + 1), (2, 2), strides=(2, 2), padding='same', trainable=trainable)(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(u8)                     # 32
    c8 = Dropout(0.1)(c8, training=uncertainty)
    c8 = Conv2D(2 ** (b + 1), (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(c8)                     # 34

    u9 = Conv2DTranspose(2 ** b, (2, 2), strides=(2, 2), padding='same', trainable=trainable)(c8)                                                    # 35
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(u9)                           # 37
    c9 = Dropout(0.1)(c9, training=uncertainty)
    c9 = Conv2D(2 ** b, (3, 3), activation='elu', kernel_initializer='he_normal', padding='same', trainable=trainable)(c9)                           # 39

    # o = Conv2D(1, (1, 1), activation='sigmoid')(c9)
    o = Conv2D(2, (1, 1), activation='softmax')(c9)

    model = Model(inputs=inputs, outputs=o)

    return model