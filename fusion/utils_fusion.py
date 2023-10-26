import re
import os
import numpy as np
import tensorflow as tf
import json
from unipath import Path
from sklearn.preprocessing import MinMaxScaler

import tensorflow.keras.backend as k
from tensorflow.keras.preprocessing.image import Iterator, ImageDataGenerator
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import model_from_json

import img_utils

offset = 6  # This is more or less 1/3s in the future!


class DroneDataGenerator(ImageDataGenerator):
    """
    Generate mini batches of images and labels with real-time augmentation.

    The only function that changes w.r.t. parent class is the flow that
    generates data. This function needed in fact adaptation for different
    directory structure and labels. All the remaining functions remain
    unchanged.

    For an example usage, see the evaluate.py script
    """

    def flow_from_directory(self, directory, aps_frame='aps', dvs_frame='dvs', is_training=False,
                            target_size=(224, 224), crop_size=(None, None), batch_size=32,
                            shuffle=True, seed=None, follow_links=False):
        return DroneDirectoryIterator(
            directory, self, aps_frame=aps_frame, dvs_frame=dvs_frame, is_training=is_training,
            target_size=target_size, crop_size=crop_size,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            follow_links=follow_links)


class DroneDirectoryIterator(Iterator):
    """
    Class for managing data loading.of images and labels
    We assume that the folder structure is:
    root_folder/
           folder_1/
                    dvs/ aps/
                    sync_steering
           folder_2/
                    dvs/ aps/
                    sync_steering
           .
           .
           folder_n/
                    dvs/ aps/
                    sync_steering

    # Arguments
       directory: Path to the root directory to read data from.
       image_data_generator: Image Generator.
       aps_frame: "aps". Frame mode to read images.
       dvs_frame: "dvs"Frame mode to read images.
       target_size: tuple of integers, dimensions to resize input images to.
       crop_size: tuple of integers, dimensions to crop input images.
       batch_size: The desired batch size
       shuffle: Whether to shuffle data or not
       seed : numpy seed to shuffle data
       follow_links: Bool, whether to follow symbolic links or not

    # TODO: Add functionality to save images to have a look at the augmentation
    """

    def __init__(self, directory, image_data_generator, aps_frame='aps', dvs_frame='dvs',
                 target_size=(224, 224), crop_size=(None, None), is_training=False,
                 batch_size=32, shuffle=True, seed=None, follow_links=False):
        self.directory = os.path.realpath(directory)
        self.image_data_generator = image_data_generator
        self.target_size = tuple(target_size)
        self.is_training = is_training
        self.crop_size = tuple(crop_size)
        self.follow_links = follow_links
        if aps_frame not in {'aps'}:
            raise ValueError('Invalid frame mode:', aps_frame, '; expected "aps".')
        if dvs_frame not in {'dvs'}:
            raise ValueError('Invalid frame mode:', dvs_frame, ';expected "dvs".')
        self.aps_frame = aps_frame
        self.dvs_frame = dvs_frame

        # Input image channels
        img_channels = 3

        # TODO: if no target size is provided, it should read image dimension
        self.image_shape = self.target_size + (img_channels,)

        # First count how many experiments are out there
        self.samples = 0

        experiments = []
        for subdir in sorted(os.listdir(directory)):
            if os.path.isdir(os.path.join(directory, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)
        self.formats = {'png', 'jpg'}

        # Idea = associate each filename with corresponding ground truths
        # (multiple predictions)
        self.filenames_aps = []
        self.filenames_dvs = []
        self.outputs = []
        self.dump_outputs = []

        for subdir in experiments:
            subpath = os.path.join(directory, subdir)
            try:
                self._decode_experiment_dir(subpath)
            except:
                continue
        if self.samples == 0:
            raise IOError("Did not find any data")

        # Conversion of list into array
        self.outputs = np.array(self.outputs, dtype=k.floatx())
        self.outputs = np.expand_dims(self.outputs, axis=-1)

        self.dump_outputs = np.array(self.dump_outputs, dtype=k.floatx())
        self.dump_outputs = np.expand_dims(self.dump_outputs, axis=-1)

        # Output dimension
        self.output_dim = self.outputs.shape[-1]

        # Steering normalization
        self.outputs = self._output_normalization(self.outputs)
        self.dump_outputs = self._output_normalization(self.dump_outputs)

        print('Found {} images belonging to {} experiments.'.format(self.samples, self.num_experiments))

        super(DroneDirectoryIterator, self).__init__(self.samples, batch_size, shuffle, seed)

        self.event_percentiles_aps = None
        self.event_percentiles_dvs = np.loadtxt(os.path.join(Path(self.directory).parent, 'percentiles.txt'), usecols=0,
                                                skiprows=1)

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=self.follow_links), key=lambda tpl: tpl[0])

    def _decode_experiment_dir(self, dir_subpath):
        # Load steerings from the experiment dir
        steerings_filename = os.path.join(dir_subpath, "sync_steering.txt")
        try:
            outputs = np.loadtxt(steerings_filename, delimiter=',', skiprows=1)
        except:
            raise IOError("Steering file not found")

        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "aps")

        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files, key=lambda fname: int(re.search(r'\d+', fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                gt_number = frame_number + offset
                for extension in self.formats:
                    if gt_number >= outputs.shape[0]:
                        break

                    if fname.lower().endswith('.' + extension):
                        # Filter those images whose velocity is under 23 km/h (for training)
                        if self.is_training:
                            if np.abs(outputs[frame_number][3]) < 2.3e1:
                                break
                            else:
                                is_valid = True
                        # Filter those images whose velocity is under 15 km/h (for evaluation)
                        else:
                            if np.abs(outputs[frame_number][3]) < 1.5e1:
                                break
                            else:
                                is_valid = True

                        # Filter 30% of images whose steering is under 5 (only for training)
                        if self.is_training:
                            if np.abs(outputs[gt_number][0]) < 5.0:
                                if np.random.random() > 0.3:
                                    is_valid = False
                                    break
                            else:
                                break

                if is_valid:
                    main_dir_path = os.path.dirname(root)
                    absolute_path_aps = os.path.join(main_dir_path, self.aps_frame, fname)
                    absolute_path_dvs = os.path.join(main_dir_path, self.dvs_frame, fname)
                    self.filenames_aps.append(os.path.relpath(absolute_path_aps, self.directory))
                    self.filenames_dvs.append(os.path.relpath(absolute_path_dvs, self.directory))
                    self.outputs.append(outputs[gt_number, 0])
                    self.dump_outputs.append(outputs[frame_number, 0])
                    self.samples += 1

    def _output_normalization(self, outputs):
        """
        Normalize input array between -1 and 1.

        # Arguments
            array: input array.
            directory:

        # Returns
            array: normalized array.
        """
        out_path = Path(self.directory).parent
        dict_path = os.path.join(out_path, 'scalers_dict.json')

        if self.is_training:
            means = np.mean(outputs)
            stds = np.std(outputs)
            # 3sigma clipping
            outputs = np.clip(outputs, means - 3 * stds, means + 3 * stds)

            # Scaling of all values
            scaler = MinMaxScaler((-1.0, 1.0))
            outputs = scaler.fit_transform(outputs)

            out_dict = {}
            out_dict['means'] = means.tolist()
            out_dict['stds'] = stds.tolist()
            out_dict['mins'] = scaler.data_min_.tolist()
            out_dict['maxs'] = scaler.data_max_.tolist()

            # Save dictionary for later testing
            with open(dict_path, 'w') as f:
                json.dump(out_dict, f)

        else:
            # Read dictionary
            with open(dict_path, 'r') as f:
                train_dict = json.load(f)

            # 3sigma clipping
            means = train_dict['means']
            stds = train_dict['stds']
            outputs = np.clip(outputs, means - 3 * stds, means + 3 * stds)

            # Scaling of all values
            mins = np.array(train_dict['mins'])
            maxs = np.array(train_dict['maxs'])

            # Range of the transformed data
            min_bound = -1.0
            max_bound = 1.0

            outputs = (outputs - mins) / (maxs - mins)
            outputs = outputs * (max_bound - min_bound) + min_bound

        return outputs

    def _get_batches_of_transformed_samples(self, index_array):
        # Make sure there are no incompatibilities with shapes
        if index_array.shape[0] != self.batch_size:
            print(f"\nIndex Array dropped\n")
            return None

        current_batch_size = index_array.shape[0]
        # Image transformation is not under thread lock, so it can be done in parallel
        batch_aps = np.zeros((current_batch_size,) + self.image_shape, dtype=k.floatx())
        batch_dvs = np.zeros((current_batch_size,) + self.image_shape, dtype=k.floatx())

        batch_outputs = np.zeros((current_batch_size, self.output_dim), dtype=k.floatx())

        # Build batch of image data for both types of frames
        for i, j in enumerate(index_array):

            fname_aps = self.filenames_aps[j]
            fname_dvs = self.filenames_dvs[j]

            aps_image = img_utils.load_img(os.path.join(self.directory, fname_aps),
                                           percentiles=self.event_percentiles_aps,
                                           frame_mode=self.aps_frame,
                                           target_size=self.target_size,
                                           crop_size=self.crop_size)
            dvs_image = img_utils.load_img(os.path.join(self.directory, fname_dvs),
                                           percentiles=self.event_percentiles_dvs,
                                           frame_mode=self.dvs_frame,
                                           target_size=self.target_size,
                                           crop_size=self.crop_size)

            # Data Augmentation for APS Frames
            if self.is_training:
                self.image_data_generator.rotation_range = 0.2
                self.image_data_generator.width_shift_range = 0.2
                self.image_data_generator.height_shift_range = 0.2
            self.image_data_generator.rescale = 1. / 255

            aps_image = self.image_data_generator.random_transform(aps_image)
            aps_image = self.image_data_generator.standardize(aps_image)

            # Data Augmentation for DVS Frames
            self.image_data_generator.rotation_range = 0.
            self.image_data_generator.rescale = None
            self.image_data_generator.width_shift_range = 0.
            self.image_data_generator.height_shift_range = 0.

            dvs_image = self.image_data_generator.random_transform(dvs_image)
            dvs_image = self.image_data_generator.standardize(dvs_image)

            batch_aps[i] = aps_image
            batch_dvs[i] = dvs_image

        # Build batch of steerings
        batch_outputs = np.array(self.outputs[index_array], dtype=k.floatx())

        # Plot the generated images
        # fig = plt.figure()
        # plt.axis('off')
        # for i in range(batch_aps.shape[0]):
        #     aps_images = batch_aps[i]
        #     dvs_images = batch_dvs[i]
        #     steerings = batch_outputs[i]
        #
        #     plt.title("Steering : {}".format(steerings))
        #     fig.add_subplot(1, 2, 1)
        #     plt.imshow(aps_images)
        #     plt.title(self.aps_frame)
        #     fig.add_subplot(1, 2, 2)
        #     plt.imshow(dvs_images)
        #     plt.title(self.dvs_frame)
        #     plt.show()
        # plt.close(fig)

        return [batch_aps, batch_dvs], batch_outputs

    def next(self):
        """
        Public function to fetch next batch
        # Returns
            The next batch of images and commands.
        """
        with self.lock:
            index_array = next(self.index_generator)

        return self._get_batches_of_transformed_samples(index_array)


def compute_predictions_and_gt(model, generator, steps,
                               max_q_size=10,
                               pickle_safe=False, verbose=0):
    """
    Generate predictions and associated ground truth
    for the input samples from a data generator.
    The generator should return the same kind of data as accepted by
    `predict_on_batch`.
    Function adapted from keras `predict_generator`.

    # Arguments
        generator: Generator yielding batches of input samples.
        steps: Total number of steps (batches of samples)
            to yield from `generator` before stopping.
        max_q_size: Maximum size for the generator queue.
        pickle_safe: If `True`, use process based threading.
            Note that because
            this implementation relies on multiprocessing,
            you should not pass
            non pick-able arguments to the generator
            as they can't be passed
            easily to children processes.
        verbose: verbosity mode, 0 or 1.

    # Returns
        Numpy array(s) of predictions and associated ground truth.

    # Raises
        ValueError: In case the generator yields
            data in an invalid format.
    """
    steps_done = 0
    all_outs = []
    all_steerings = []

    if verbose == 1:
        progbar = Progbar(target=steps)

    while steps_done < steps:
        generator_output = next(generator)

        if isinstance(generator_output, tuple):
            if len(generator_output) == 2:
                x, gt_steer = generator_output
            elif len(generator_output) == 3:
                x, gt_steer, _ = generator_output
            else:
                raise ValueError('output of generator should be '
                                 'a tuple `(x, y, sample_weight)` '
                                 'or `(x, y)`. Found: ' +
                                 str(generator_output))
        else:
            raise ValueError('Output not valid for current evaluation')

        outs = model.predict_on_batch(x)
        # outs = gt_steer

        if not isinstance(outs, list):
            outs = [outs]
        if not isinstance(gt_steer, list):
            gt_steer = [gt_steer]

        if not all_outs:
            for out in outs:
                # Len of this list is related to the number of outputs per model(1 in our case)
                all_outs.append([])

        if not all_steerings:
            # Len of list related to the number of gt_steerings per model (1 in our case )
            for steer in gt_steer:
                all_steerings.append([])

        for i, out in enumerate(outs):
            all_outs[i].append(out)

        for i, steer in enumerate(gt_steer):
            all_steerings[i].append(steer)

        steps_done += 1
        if verbose == 1:
            progbar.update(steps_done)

    if steps_done == 1:
        return [out for out in all_outs], [steer for steer in all_steerings]
    else:
        return np.squeeze(np.array([np.concatenate(out) for out in all_outs])), \
            np.squeeze(np.array([np.concatenate(steer) for steer in all_steerings]))


def hard_mining_mse(k):
    """
    Compute MSE for steering evaluation and hard-mining for the current batch.

    # Arguments
        k: number of samples for hard-mining.

    # Returns
        custom_mse: average MSE for the current batch.
    """

    def custom_mse(y_true, y_pred):
        # Steering loss
        l_steer = tf.keras.backend.square(y_pred - y_true)
        l_steer = tf.squeeze(l_steer, squeeze_dims=-1)

        # Hard mining
        k_min = tf.minimum(k, tf.shape(l_steer)[0])
        _, indices = tf.nn.top_k(l_steer, k=k_min)
        max_l_steer = tf.gather(l_steer, indices)
        hard_l_steer = tf.divide(tf.reduce_sum(max_l_steer), tf.cast(k, tf.float32))

        return hard_l_steer

    return custom_mse


def steering_loss(y_true, y_pred):
    return tf.reduce_mean(k.square(y_pred - y_true))


def pred_std(y_true, y_pred):
    _, var = tf.nn.moments(y_pred, axes=[0])
    return tf.sqrt(var)


def modelToJson(model, json_model_path):
    """
    Serialize model into json.
    """
    model_json = model.to_json()

    with open(json_model_path, "w") as f:
        f.write(model_json)


def jsonToModel(json_model_path):
    """
    Serialize json into model.
    """
    with open(json_model_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    model = model_from_json(loaded_model_json)

    return model


def write_to_file(dictionary, fname):
    """
    Writes everything is in a dictionary in json model.
    """
    with open(fname, "w") as f:
        json.dump(dictionary, f)
        print("Written file {}".format(fname))
