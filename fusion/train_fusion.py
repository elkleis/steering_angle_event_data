"""
Script for training model.
"""

import numpy as np
import os
import sys
import gflags
import glob
import shutil

import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as k

from intermediate_fusion import intermediate_fusion, fusion_identity
from common_flags_fusion import FLAGS
import logz
import utils_fusion as utils
import log_utils
from constants import TRAIN_PHASE

tf.keras.backend.set_floatx('float32')


def get_model(img_width, img_height, img_channels, output_dim, weights_path, batch_size):
    """
    Initialize model.
    """
    # model = self_attention_model(img_width, img_height, img_channels, batch_size)
    model = intermediate_fusion(img_width, img_height, img_channels, output_dim, batch_size)

    if weights_path:
        model.load_weights(weights_path)
        print("Loaded model from {}".format(weights_path))

    return model


def train_model(train_generator, validation_generator, model, initial_epoch, weights_dir):
    """
    Model training.

    # Arguments
       train_generator: Training dvs and aps data generated batch by batch.
       validation_generator: Validation dvs and aps data generated batch by batch.
       model: Target image channels.
       initial_epoch: Dimension of model output.
       weights_dir : Directory to save model weights.
    """

    # Initialize number of samples for hard-mining
    model.k_mse = tf.Variable(FLAGS.batch_size, trainable=False, name='k_mse', dtype=tf.int32)

    # Initialize variables
    init = tf.global_variables_initializer()
    k.get_session().run(init)

    # Configure training process
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.initial_lr, decay=1e-4)
    model.compile(loss=[utils.hard_mining_mse(model.k_mse)], optimizer=optimizer,
                  metrics=[utils.steering_loss, utils.pred_std])

    # Save model with the lowest validation loss
    weights_path = os.path.join(weights_dir, 'weights_{epoch:03d}.h5')
    best_weights = os.path.join(weights_dir, 'best_weights.h5')
    write_best_model = ModelCheckpoint(filepath=weights_path, monitor='val_steering_loss', save_best_only=True,
                                       save_weights_only=True)

    # Save training and validation losses every 'log_rate' epochs
    logz.configure_output_dir(FLAGS.experiment_rootdir)
    save_model_and_loss = log_utils.MyCallback(filepath=FLAGS.experiment_rootdir,
                                               period=FLAGS.log_rate,
                                               batch_size=FLAGS.batch_size,
                                               factor=FLAGS.lr_scale_factor)

    # Stop training if there is no improvement for 20 epochs
    early_stopping = EarlyStopping(monitor='val_steering_loss', patience=20)

    # Train model
    train_samples = train_generator.samples
    steps_per_epoch = np.minimum(int(np.ceil(train_samples / FLAGS.batch_size)), 2000)
    validation_steps = int(np.ceil(train_samples / FLAGS.batch_size)) - 1

    model.fit_generator(train_generator,
                        epochs=FLAGS.epochs, steps_per_epoch=steps_per_epoch,
                        callbacks=[write_best_model, save_model_and_loss, early_stopping],
                        validation_data=validation_generator,
                        validation_steps=validation_steps,
                        initial_epoch=initial_epoch)

    files = glob.glob(os.path.join(weights_dir, 'weights_*.h5'))
    src = sorted(files)[-1]
    shutil.copyfile(src, best_weights)


def _main():
    # Output dimension for steering angle
    output_dim = 1
    # Image channels for both dvs and aps frames
    img_channel = 3

    # Set random seed
    if FLAGS.random_seed:
        seed = np.random.randint(0, 2 * 31 - 1)
    else:
        seed = 5
    np.random.seed(seed)
    tf.set_random_seed(seed)

    k.set_learning_phase(TRAIN_PHASE)

    # Create the experiment rootdir if not already there
    if not os.path.exists(FLAGS.experiment_rootdir):
        os.makedirs(FLAGS.experiment_rootdir)

    # Generate training data with real-time augmentation
    train_datagen = utils.DroneDataGenerator()
    train_generator = train_datagen.flow_from_directory(FLAGS.train_dir,
                                                        is_training=True,
                                                        shuffle=True,
                                                        aps_frame="aps",
                                                        dvs_frame="dvs",
                                                        target_size=(FLAGS.img_height, FLAGS.img_width),
                                                        crop_size=(FLAGS.crop_img_height, FLAGS.crop_img_width),
                                                        batch_size=FLAGS.batch_size)

    # Generate training data with real-time augmentation
    validation_datagen = utils.DroneDataGenerator()
    validation_generator = validation_datagen.flow_from_directory(FLAGS.train_dir,
                                                                  is_training=False,
                                                                  shuffle=False,
                                                                  aps_frame="aps",
                                                                  dvs_frame="dvs",
                                                                  target_size=(FLAGS.img_height, FLAGS.img_width),
                                                                  crop_size=(
                                                                  FLAGS.crop_img_height, FLAGS.crop_img_width),
                                                                  batch_size=FLAGS.batch_size)

    assert train_generator.output_dim == validation_generator.output_dim, \
        " Not matching output dimensions."
    output_dim = train_generator.output_dim

    # Weights to restore
    weights_dir = os.path.join(FLAGS.experiment_rootdir, 'weights')
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    model_weights_dir = os.path.join(weights_dir, 'epochs_period')
    if not os.path.exists(model_weights_dir):
        os.makedirs(model_weights_dir)
    weights_path = os.path.join(weights_dir, FLAGS.weights_fname)
    initial_epoch = 0
    if not FLAGS.restore_model:
        # In this case weights will start from random
        weights_path = None
    else:
        # In this case weights will start from the specified model
        initial_epoch = FLAGS.initial_epoch

    # Define model
    model = get_model(FLAGS.img_width, FLAGS.img_height, img_channel, output_dim, weights_path, FLAGS.batch_size)

    # Serialize model into json
    json_model_path = os.path.join(FLAGS.experiment_rootdir, FLAGS.json_model_fname)
    utils.modelToJson(model, json_model_path)

    # Train model
    train_model(train_generator, validation_generator, model, initial_epoch, weights_dir)


def main(argv):
    # Utility main to load flags
    try:
        argv = FLAGS(argv)  # parse flags
    except gflags.FlagsError:
        print('Usage: %s ARGS\\n%s' % (sys.argv[0], FLAGS))

        sys.exit(1)
    _main()


if __name__ == "__main__":
    main(sys.argv)
