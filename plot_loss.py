"""
Creates plot for training/validation loss - epochs.
"""

import os
import sys
import numpy as np
import gflags
import matplotlib.pyplot as plt
import matplotlib

matplotlib.style.use('ggplot')

from common_flags import FLAGS

gflags.DEFINE_string("exp_root_2", "../training/", "Folder where to take the second experiment")


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth


def _main():
    # Read log file
    log_files = [os.path.join(FLAGS.experiment_rootdir, "log.txt")]
    logs = []
    for log_file in log_files:
        try:
            logs.append(np.genfromtxt(log_file, delimiter='\t', dtype=None, names=True))
        except:
            raise IOError("Log file not found")

    train_loss_1 = logs[0]['steering_loss']
    train_loss_1 = smooth(train_loss_1, 10)
    timesteps = list(range(train_loss_1.shape[0]))

    # Plot losses
    fig = plt.figure(1, figsize=(17, 8))
    ax = fig.add_subplot(111)
    ax.plot(timesteps, timesteps, train_loss_1, 'b', linewidth=7)
    plt.legend(["Random Initialization", "ImageNet Initialization"], fontsize=40)
    plt.ylabel('Loss', size=45)
    plt.xlabel('Epoch', size=45)
    plt.yscale('log')
    plt.tick_params(labelsize=35)
    plt.savefig(os.path.join(FLAGS.experiment_rootdir, "plots", "log.png"), bbox_inches='tight')


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
