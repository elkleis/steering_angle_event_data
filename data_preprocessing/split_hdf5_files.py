"""
Splits files with size larger than 2.7GB for proper pre-processing

Ussage
$ ./split_hdf5_files.py --parent_folder <directory> --in_file <file_to_split>
"""

import argparse
import os
import h5py

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent_folder', default='/mnt/seagate/DDD17/ddd17/not_used_ddd17/frames_too_long',
                        help='The parent folder where you can find all the files sub-folders')
    parser.add_argument('--in_file', default='rec1487417411', help='File to split.')
    args = parser.parse_args()

    print('Splitting file {}...'.format(args.in_file))
    # Check if split directory exists
    if not os.path.exists(os.path.join(args.parent_folder, args.in_file, 'split')):
        os.makedirs(os.path.join(args.parent_folder, args.in_file, 'split'))

    input_file = os.path.join(args.parent_folder, args.in_file, args.in_file + '.hdf5')
    output_file1 = os.path.join(args.parent_folder, args.in_file, 'split', args.in_file + '1.hdf5')
    output_file2 = os.path.join(args.parent_folder, args.in_file, 'split', args.in_file + '2.hdf5')

    # Check whether the split files still exist, and delete them
    if os.path.exists(output_file1):
        os.remove(output_file1)
    if os.path.exists(output_file2):
        os.remove(output_file2)

    # Read original file and create the new split ones
    infile = h5py.File(input_file, 'r')
    outfile1 = h5py.File(output_file1, 'w')
    outfile2 = h5py.File(output_file2, 'w')

    # Read all the datasets in the original HDF5 file
    for dataset_name in infile:
        dataset = infile[dataset_name]

        # Find size of dataset
        size = []
        size2 = []
        for i in range(0, len(dataset.shape)):
            if i == 0:
                shape_zero = int(dataset.shape[i] / 2)
                size.append(shape_zero)
                size2.append(shape_zero)
            else:
                size.append(int(dataset.shape[i]))
                size2.append(int(dataset.shape[i]))

        if (size2[0] % 2) != 0:
            size2[0] = size2[0] + 1
        size = tuple(size)
        size2 = tuple(size2)

        # Copy the first half data of original file to first split file
        out_dataset1 = outfile1.create_dataset(dataset_name, shape=size, dtype=dataset.dtype, compression='gzip',
                                               compression_opts=4)
        out_dataset1[:] = dataset[:shape_zero]

        # Copy the second half data of original file to second split file
        out_dataset2 = outfile2.create_dataset(dataset_name, shape=size2, dtype=dataset.dtype, compression='gzip',
                                               compression_opts=4)
        out_dataset2[:] = dataset[shape_zero:]

    # Close all file
    infile.close()
    outfile1.close()
    outfile2.close()

    print('Done splitting...')
