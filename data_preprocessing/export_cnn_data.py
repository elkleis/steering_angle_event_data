"""
Export DVS frames, APS frames and steering angles to be used by
the networks.
"""

import h5py
import numpy as np
import cv2
import os
import argparse
import glob
from itertools import groupby
from operator import itemgetter
import shutil


def split_sequence(data):
    sequences = []
    for k, g in groupby(enumerate(data), lambda (i, x): i - x):
        sequences.append(map(itemgetter(1), g))
    return sequences


def export_data(f_in, idxs, out_path, pos_inf, pos_sup, neg_inf, neg_sup, rotate180):
    # Non-image data
    data = np.zeros((len(idxs), 4))

    for key in f_in.keys():
        key = str(key)

        # Export DVS frames
        if key == 'dvs_frame':
            dvs_path = os.path.join(out_path, 'dvs')
            # Delete previous images
            if os.path.exists(dvs_path):
                shutil.rmtree(dvs_path)
            os.makedirs(dvs_path)

            images = f_in[key][idxs]
            for i in range(images.shape[0]):
                new_img = np.zeros((images.shape[1], images.shape[2], 3), dtype=np.uint8)
                event_img = images[i]

                # Positive events to channel 0
                pos_img = event_img[:, :, 0]
                index_p = pos_img > 0
                pos_img[index_p] = np.clip(pos_img[index_p], pos_inf, pos_sup)
                new_img[:, :, 0] = pos_img

                # Negative events to channel 1
                neg_img = event_img[:, :, 1]
                index_n = neg_img > 0
                neg_img[index_n] = np.clip(neg_img[index_n], neg_inf, neg_sup)
                new_img[:, :, -1] = neg_img

                # Rotate image by 180 degrees if needed
                if rotate180:
                    (h, w) = new_img.shape[:2]
                    center = (w / 2, h / 2)
                    m = cv2.getRotationMatrix2D(center, 180, 1.0)
                    new_img = cv2.warpAffine(new_img, m, (w, h))

                # Save DVS frame
                img_name = "frame_" + str(i).zfill(5) + ".png"
                cv2.imwrite(os.path.join(dvs_path, img_name), new_img)

        # Export APS frames
        elif key == 'aps_frame':
            aps_path = os.path.join(out_path, 'aps')
            if os.path.exists(aps_path):
                shutil.rmtree(aps_path)
            os.makedirs(aps_path)

            images = f_in[key][idxs]
            images = np.asarray(images, dtype=np.uint8)
            for i in range(images.shape[0]):
                img_name = "frame_" + str(i).zfill(5) + ".png"
                cv2.imwrite(os.path.join(aps_path, img_name), images[i, :, :])

                # Save APS frames and rotate image by 180 degrees if needed
                if rotate180:
                    img = images[i, :, :]
                    (h, w) = img.shape[:2]
                    center = (w / 2, h / 2)
                    M = cv2.getRotationMatrix2D(center, 180, 1.0)
                    img = cv2.warpAffine(img, M, (w, h))
                    cv2.imwrite(os.path.join(aps_path, img_name), img)
                else:
                    cv2.imwrite(os.path.join(aps_path, img_name), images[i, :, :])

        # Steering, torque, engine speed, vehicle speed associated to DVS and APS frames
        elif key == 'steering_wheel_angle':
            steer = f_in[key][idxs]
            data[:, 0] = steer
        elif key == 'torque_at_transmission':
            torque = f_in[key][idxs]
            data[:, 1] = torque
        elif key == 'engine_speed':
            eng_speed = f_in[key][idxs]
            data[:, 2] = eng_speed
        elif key == 'vehicle_speed':
            veh_speed = f_in[key][idxs]
            data[:, 3] = veh_speed

    # Save steering angles
    txt_name = os.path.join(out_path, 'sync_steering.txt')
    np.savetxt(txt_name, data, delimiter=',', header='steering, torque, engine_velocity, vehicle_velocity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # DDD17
    parser.add_argument('--source_folder', default='/mnt/Seagate/DDD17/ddd17/model/data',
                        help='Path to frame-based hdf5 files.')
    parser.add_argument('--rotate', type=int, default=0,
                        help='Whether to rotate the image and steering angles. Zero(0) for DDD17, One(1) for DDD20')
    # # DDD20
    # parser.add_argument('--source_folder', default='/mnt/Seagate/DDD17/ddd20/model/data',
    #                     help='Path to frame-based hdf5 files.')
    # parser.add_argument('--rotate', type=int, default=1,
    #                     help='Whether to rotate the image and steering angles. Zero(0) for DDD17, One(1) for DDD20')
    args = parser.parse_args()

    # Load percentiles
    try:
        percentiles = np.loadtxt(os.path.join(args.source_folder, 'percentiles.txt'), usecols=0, skiprows=1)
    except:
        raise IOError("Percentiles file not found")
    pos_inf = percentiles[1]  # Inferior percentile for positive events
    pos_sup = percentiles[2]  # Superior percentile for positive events
    neg_inf = percentiles[4]  # Inferior percentile for negative events
    neg_sup = percentiles[5]  # Superior percentile for negative events

    # For every recording/hdf5 file
    recordings = glob.glob(args.source_folder + '/*.hdf5')
    for rec in recordings:
        print("Exporting frames for '{}'".format(rec))

        # Open the HDF5 file in read write mode and rotate the steering wheel angles if rotate is true
        if args.rotate:
            with h5py.File(rec, 'r+') as f:
                dataset = f['steering_wheel_angle']
                dataset[:] = - dataset[:]
            f.close()

        f_in = h5py.File(rec, 'r')
        # Name of the experiment
        exp_name = rec.split('.')[-2]
        exp_name = exp_name.split('/')[-1]

        # Get training sequences
        if 'train_idxs' in f_in:
            train_idxs = np.ndarray.tolist(f_in['train_idxs'][()])
            train_sequences = split_sequence(train_idxs)
            # Create training set
            for i, train_seq in enumerate(train_sequences):
                output_path = os.path.join(args.source_folder, 'training', exp_name + str(i))
                if not os.path.exists(output_path):
                    os.makedirs(output_path)
                export_data(f_in, train_seq, output_path, pos_inf, pos_sup, neg_inf, neg_sup, args.rotate)

        # Get testing sequences
        test_idxs = np.ndarray.tolist(f_in['test_idxs'][()])
        test_sequences = split_sequence(test_idxs)
        # Create testing set
        for j, test_seq in enumerate(test_sequences):
            output_path = os.path.join(args.source_folder, 'testing', exp_name + str(j))
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            export_data(f_in, test_seq, output_path, pos_inf, pos_sup, neg_inf, neg_sup, args.rotate)

        f_in.close()

    print("\nDone. Frames and data exported.\n")
