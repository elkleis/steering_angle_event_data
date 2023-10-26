from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, Input, Concatenate, GlobalAveragePooling2D, BatchNormalization, Activation, \
    Conv2D, add
from tensorflow.keras.models import Model


def intermediate_fusion(img_width, img_height, img_channels, output_dim, batch_size):
    # Create Input Layers
    aps_input = Input(batch_size=batch_size, shape=(img_height, img_width, img_channels))
    dvs_input = Input(batch_size=batch_size, shape=(img_height, img_width, img_channels))

    # Load ResNet50 models
    resnet_model_aps = ResNet50(input_tensor=aps_input, weights='imagenet', include_top=False)
    resnet_model_dvs = ResNet50(input_tensor=dvs_input, weights='imagenet', include_top=False)

    # resnet_model_dvs.summary()
    # resnet_model_aps.summary()

    for layer in resnet_model_dvs.layers:
        layer._name = 'dvs_' + layer.name

    aps = resnet_model_aps.output  # Last APS Feature Map
    aps = GlobalAveragePooling2D()(aps)  # Global Average Pooling on APS Feature Map
    dvs = resnet_model_dvs.output  # Last DVS Feature Map
    dvs = GlobalAveragePooling2D()(dvs)  # Global Average Pooling on DVS Feature Map

    # Fusion with concatenation of the two types of frames
    fusion = Concatenate()([aps, dvs])

    # Decoder
    # fusion = Dense(1024, input_shape=(fusion.shape[1].value,), activation='relu')(fusion)
    fusion = Dense(2048, input_shape=(fusion.shape[1].value,), activation='relu')(fusion)
    fusion = Dense(output_dim)(fusion)

    model = Model(inputs=[aps_input, dvs_input], outputs=[fusion])
    # model.summary()

    return model


def fusion_identity(img_width, img_height, img_channels, output_dim, batch_size):
    # Create Input Layers
    aps_input = Input(batch_size=batch_size, shape=(img_height, img_width, img_channels))
    dvs_input = Input(batch_size=batch_size, shape=(img_height, img_width, img_channels))

    # Load ResNet50 models
    resnet_model_aps = ResNet50(input_tensor=aps_input, weights='imagenet', include_top=False)
    resnet_model_dvs = ResNet50(input_tensor=dvs_input, weights='imagenet', include_top=False)

    # resnet_model_dvs.summary()
    # resnet_model_aps.summary()

    for layer in resnet_model_dvs.layers:
        layer._name = 'dvs_' + layer.name

    aps = resnet_model_aps.output  # Last APS Feature Map
    dvs = resnet_model_dvs.output  # Last DVS Feature Map

    # Fusion with concatenation of the two types of frames
    fusion = Concatenate()([aps, dvs])

    # Identity Block
    # First component of main path
    x = Conv2D(1024, (1, 1), kernel_initializer='he_normal', name='identity_conv_1')(fusion)
    x = BatchNormalization(axis=3, name='identity_batch_norm_1')(x)
    x = Activation('relu')(x)

    # Second component of main path
    x = Conv2D(1024, (3, 3), padding='same', kernel_initializer='he_normal', name='identity_conv_2')(x)
    x = BatchNormalization(axis=3, name='identity_batch_norm_2')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(4096, (1, 1), kernel_initializer='he_normal', name='identity_conv_3')(x)
    x = BatchNormalization(axis=3, name='identity_batch_norm_3')(x)

    # Add shortcut and main path values
    fusion = add([x, fusion])
    fusion = Activation('relu')(fusion)

    # Decoder
    fusion = GlobalAveragePooling2D()(fusion)
    fusion = Dense(1024, input_shape=(fusion.shape[1].value,), activation='relu')(fusion)
    fusion = Dense(output_dim)(fusion)

    model = Model(inputs=[aps_input, dvs_input], outputs=[fusion])
    # model.summary()

    return model