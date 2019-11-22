from keras.initializers import RandomNormal
from keras.models import Input
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Concatenate
from keras.layers import Conv2DTranspose
from keras.optimizers import Adam
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

def discriminator(image_shape):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)
    
    layer = Conv2D(64, (4,4), strides=(2,2),padding='same',kernel_initializer=init)(in_image)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(128, (4,4), strides=(2,2),padding='same',kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(256, (4,4), strides=(2,2),padding='same',kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4,4), strides=(2,2),padding='same',kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    layer = Conv2D(512, (4,4), padding='same',kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = LeakyReLU(alpha=0.2)(layer)

    patch_out = Conv2D(1,(4,4), padding='same', kernel_initializer=init)(layer)

    model = Model(in_image, patch_out)

    model.compile(loss='mse', optimizer=Adam(lr=0.0002,beta_1=0.5),loss_weights=[0.5])
    
    return model

def resnet_block(n_filters, input_layer):
    init = RandomNormal(stddev=0.02)

    layer = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(input_layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(n_filters, (3,3), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)

    layer = Concatenate()([layer, input_layer])

    return layer

def generator(image_shape, n_resnet=9):
    init = RandomNormal(stddev=0.02)
    in_image = Input(shape=image_shape)

    layer = Conv2D(64, (7,7), padding='same', kernel_initializer=init)(in_image)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(256, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    for _ in range(n_resnet):
        layer = resnet_block(256,layer)
    
    layer = Conv2DTranspose(128, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2DTranspose(64, (3,3), strides=(2,2), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    layer = Activation('relu')(layer)

    layer = Conv2D(3, (7,7), padding='same', kernel_initializer=init)(layer)
    layer = InstanceNormalization(axis=-1)(layer)
    out_image = Activation('tanh')(layer)

    model = Model(in_image, out_image)
    return model

def composite_model(g_model_1,d_model,g_model_2,image_shape):
    g_model_1.trainable = True
    d_model.trainable = False
    g_model_2.trainable = False

    input_gen = Input(shape=image_shape)
    gen1_out = g_model_1(input_gen)
    output_d = d_model(gen1_out)

    input_id = Input(shape=image_shape)
    output_id = g_model_1(input_id)

    output_f = g_model_2(gen1_out)

    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)

    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])

    opt = Adam(lr=0.0002, beta_1=0.5)

    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model