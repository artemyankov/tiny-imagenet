import os
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def read_img_dir_for_tf(data_path):
    """Extracts image filepaths and labels from Tiny ImageNet directory structure
    for use in tensorflow.
    """
    print ('Reading data directory {0}'.format(data_path))
    img_files = []
    labels = []
    label_map = {}
    label_iter = 0
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.JPEG'):
                img_files.append(
                    os.path.join(root, file)
                )
                label_key = file.split('_')[0]
                try:
                    label_val = label_map[label_key]
                except KeyError:
                    label_val = label_iter
                    label_map[label_key] = label_iter
                    label_iter += 1
                finally:
                    labels.append(label_val)
    return(img_files, labels, label_map)

def _parse_function(filename):
    """All image preprocessing functions such as resize (tf.image.resize), perturbations, ...
    """
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
    image_decoded = image_decoded / 255.0
    image_decoded.set_shape([64, 64, 3])

    return image_decoded

def input_fn(data_path, num_epochs=None, shuffle=True, batch_size=32):
    img_files, labels, _ = read_img_dir_for_tf(data_path)
    img_filenames = tf.constant(img_files)
    img_labels = tf.constant(labels)

    dataset_x = tf.data.Dataset.from_tensor_slices(img_filenames)
    dataset_x = dataset_x.map(_parse_function)
    dataset_y = tf.data.Dataset.from_tensor_slices(img_labels)
    dataset_y = dataset_y.map(lambda z: tf.one_hot(z, FLAGS.N_CLASSES, dtype=tf.int32))

    dataset = tf.data.Dataset.zip((dataset_x, dataset_y))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 10)

    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()

    return features, labels