import math
import os
import random
import sys
import glob
import tensorflow as tf
from lib import dataset_utils
_NUM_VALIDATION = 0
_RANDOM_SEED = 0
_NUM_SHARDS = 4
class_num = 19
dir_list = '*'
dataset_name = 'jp_2s'
seq = False
dataset_path = './dataset/action_data'
output_path = './dataset/action_merge'
parts = ['t2', 't3']
label_name = 'label.txt'
output_dir = os.path.join(output_path, dataset_name)
class ImageReader(object):
    """Helper class that provides TensorFlow image coding utilities."""
    def __init__(self):
        pass
    def read_image_dims(self, image_data):
        image = self.decode_jpeg(image_data)
        return image.shape[0], image.shape[1]
    def decode_jpeg(self, image_data):
        image = tf.io.decode_jpeg(image_data, channels=3)
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image

def _get_filenames_and_classes(dataset_path):
    file_list = []
    for folder in glob.glob(os.path.join(dataset_path, dir_list)):
        with open(os.path.join(folder, label_name)) as list_file:
            file_list += list_file.readlines()
    return file_list

def _get_dataset_filename(output_dir, split_name, shard_id):
    output_filename = 'clip_%s_%05d-of-%05d.tfrecord' % (split_name, shard_id, _NUM_SHARDS)
    return os.path.join(output_dir, output_filename)

def _convert_dataset(split_name, filenames, dataset_path):
    assert split_name in ['train', 'validation']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img_files_a = []
    img_files_b = []
    class_names = []

    for filename in filenames:
        img_files_a.append(os.path.join(dataset_path, filename.split('_')[0], parts[0], filename.strip().split()[0].split('_')[1]))
        img_files_b.append(os.path.join(dataset_path, filename.split('_')[0], parts[1], filename.strip().split()[0].split('_')[1]))
        if int(filename.strip().split()[1]) < class_num:
            class_names.append(filename.strip().split()[1])
        else:
            print(filename)
            exit(0)

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(output_dir, split_name, shard_id)

        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id + 1) * num_per_shard, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (i + 1, len(filenames), shard_id))
                sys.stdout.flush()

                image_a = tf.io.read_file(img_files_a[i])
                image_b = tf.io.read_file(img_files_b[i])
                class_id = int(class_names[i])
                example = dataset_utils.image_to_tfexample(image_a, image_b, b'jpg', class_id)
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()

def main():
    file_list = _get_filenames_and_classes(dataset_path)
    if seq:
        file_list.sort()
        training_filenames = file_list[_NUM_VALIDATION:]
    else:
        random.seed(_RANDOM_SEED)
        random.shuffle(file_list)
    training_filenames = file_list[_NUM_VALIDATION:]
    _convert_dataset('train', training_filenames, dataset_path)
    print('\nFinished converting the dataset!')

if __name__ == '__main__':
    main()