from keras import Model
from keras.layers import GlobalAveragePooling2D, Dense, Concatenate, Conv2D, BatchNormalization, Activation, add
from classification_models.keras import Classifiers


def intermediate_fusion_resnet34(img_height, img_width, img_channels, output_dim):
    input_shape = (img_height, img_width, img_channels)

    # Load pre-trained ResNet34 models(available from https://github.com/qubvel/classification_models)
    pretrained_resnet34, preprocess_input = Classifiers.get('resnet34')
    aps_model = pretrained_resnet34(input_shape=input_shape, weights='imagenet', include_top=False)
    dvs_model = pretrained_resnet34(input_shape=input_shape, weights='imagenet', include_top=False)

    for layer in dvs_model.layers:
        layer.name = 'dvs_' + layer.name

    # aps_model.summary()
    # dvs_model.summary()

    aps_input = aps_model.input
    dvs_input = dvs_model.input
    aps = aps_model.output  # Last APS Feature Map
    aps = GlobalAveragePooling2D()(aps)  # Global Average Pooling on APS Feature Map
    dvs = dvs_model.output  # Last DVS Feature Map
    dvs = GlobalAveragePooling2D()(dvs)  # Global Average Pooling on DVS Feature Map

    # Fusion with concatenation of the two types of frames
    fusion = Concatenate()([aps, dvs])

    # Decoder
    fusion = Dense(1024, input_shape=(fusion.shape[1].value,), activation='relu')(fusion)
    fusion = Dense(output_dim)(fusion)

    model = Model(inputs=[aps_input, dvs_input], outputs=[fusion])
    model.summary()

    return model


def fusion_identity_resnet34(img_width, img_height, img_channels, output_dim):
    input_shape = (img_height, img_width, img_channels)

    # Load pre-trained ResNet34 models(available from https://github.com/qubvel/classification_models)
    pretrained_resnet34, preprocess_input = Classifiers.get('resnet34')
    aps_model = pretrained_resnet34(input_shape=input_shape, weights='imagenet', include_top=False)
    dvs_model = pretrained_resnet34(input_shape=input_shape, weights='imagenet', include_top=False)

    # aps_model.summary()
    # dvs_model.summary()

    for layer in dvs_model.layers:
        layer.name = 'dvs_' + layer.name

    aps = aps_model.output  # Last APS Feature Map
    dvs = dvs_model.output  # Last DVS Feature Map

    # Fusion with concatenation of the two types of frames
    fusion = Concatenate()([aps, dvs])

    # Identity Block
    # First component of main path
    x = Conv2D(256, (1, 1), kernel_initializer='he_normal', name='identity_conv_1')(fusion)
    x = BatchNormalization(axis=3, name='identity_batch_norm_1')(x)
    x = Activation('relu')(x)

    # Second component of main path KERNEL SIZE
    x = Conv2D(256, (3, 3), padding='same', kernel_initializer='he_normal', name='identity_conv_2')(x)
    x = BatchNormalization(axis=3, name='identity_batch_norm_2')(x)
    x = Activation('relu')(x)

    # Third component of main path
    x = Conv2D(1024, (1, 1), kernel_initializer='he_normal', name='identity_conv_3')(x)
    x = BatchNormalization(axis=3, name='identity_batch_norm_3')(x)

    # Add shortcut and main path values
    fusion = add([x, fusion])
    fusion = Activation('relu')(fusion)

    # Decoder
    fusion = GlobalAveragePooling2D()(fusion)
    fusion = Dense(1024, input_shape=(fusion.shape[1].value,), activation='relu')(fusion)
    fusion = Dense(output_dim)(fusion)

    model = Model(inputs=[aps_model.input, dvs_model.input], outputs=[fusion])
    model.summary()

    return model
