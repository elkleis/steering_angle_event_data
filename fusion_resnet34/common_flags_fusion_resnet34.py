import gflags

FLAGS = gflags.FLAGS

# Random seed
gflags.DEFINE_bool('random_seed', True, 'Random seed')

# Input
gflags.DEFINE_integer('img_width', 200, 'Target Image Width')
gflags.DEFINE_integer('img_height', 200, 'Target Image Height')

gflags.DEFINE_integer('crop_img_width', 200, 'Cropped image width')
gflags.DEFINE_integer('crop_img_height', 200, 'Cropped image height')

gflags.DEFINE_string('frame_mode', "fusion", 'Load mode for images, either dvs, aps or aps_diff or fusion')
gflags.DEFINE_string('visual_mode', "grayscale", 'Mode for video visualization, either red_blue or grayscale')

# Training
gflags.DEFINE_integer('batch_size', 32, 'Batch size in training and evaluation')
gflags.DEFINE_integer('epochs', 200, 'Number of epochs for training')
gflags.DEFINE_integer('log_rate', 10, 'Logging rate for full model (epochs)')
gflags.DEFINE_integer('initial_epoch', 0, 'Initial epoch to start training')
gflags.DEFINE_float('initial_lr', 1e-3, 'Initial learning rate for adam')
gflags.DEFINE_float('lr_scale_factor', 0.5, 'Reducer factor for learning rate when loss stagnates.')

# Files
gflags.DEFINE_string('experiment_rootdir', "/mnt/Seagate/DDD17/fusion/fusion",
                     'Folder containing all the logs, model weights and results')
gflags.DEFINE_string('train_dir', "/mnt/Seagate/DDD17/fusion/data/training",
                     'Folder containing training experiments')
gflags.DEFINE_string('val_dir', "/mnt/Seagate/DDD17/fusion/data/validation",
                     'Folder containing validation experiments')
gflags.DEFINE_string('test_dir', "/mnt/Seagate/DDD17/fusion/data/testing",
                     'Folder containing testing experiments')
gflags.DEFINE_string('video', "rec14873264220", 'Video processed on its own')
gflags.DEFINE_string('video_dir', "/mnt/Seagate/DDD17/fusion/fusion/videos/process_videos/rec14873264220",
                     'Directory of single video to process')

# Model
gflags.DEFINE_bool('restore_model', False, 'Whether to restore a trained model for training')
gflags.DEFINE_string('weights_fname', "best_weights.h5",
                     'Filename of model weights, usually the weights file from the last epoch')
gflags.DEFINE_string('json_model_fname', "model_struct.json", 'Model struct json serialization, filename')
