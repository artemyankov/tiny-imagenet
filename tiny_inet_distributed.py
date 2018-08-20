"""
"""
import tensorflow as tf
import keras
import os
from collections import namedtuple

#
# Command line arguments
#
N_CLASSES = 200
BATCH_SIZE = 32
SEED = 45334
VAL_FREQUENCY_STEPS = 500

# ----- Insert that snippet to run distributed jobs -----

from clusterone import get_data_path, get_logs_path

# Specifying paths when working locally
# For convenience we use a clusterone wrapper (get_data_path below) to be able
# to switch from local to clusterone without cahnging the code.

# Configure  distributed task
try:
  job_name = os.environ['JOB_NAME']
  task_index = os.environ['TASK_INDEX']
  ps_hosts = os.environ['PS_HOSTS']
  worker_hosts = os.environ['WORKER_HOSTS']
except:
  job_name = None
  task_index = 0
  ps_hosts = None
  worker_hosts = None

flags = tf.app.flags

# Flags for configuring the distributed task
flags.DEFINE_string("job_name", job_name,
                    "job name: worker or ps")
flags.DEFINE_integer("task_index", task_index,
                     "Worker task index, should be >= 0. task_index=0 is "
                     "the chief worker task that performs the variable "
                     "initialization and checkpoint handling")
flags.DEFINE_string("ps_hosts", ps_hosts,
                    "Comma-separated list of hostname:port pairs")
flags.DEFINE_string("worker_hosts", worker_hosts,
                    "Comma-separated list of hostname:port pairs")

# Training related flags
flags.DEFINE_string("train_data_dir",
                    get_data_path(
                        dataset_name = 'artem/artem-tiny-imagenet',
                        local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
                        local_repo = 'tiny-imagenet-200',
                        path = 'train'
                    ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("val_data_dir",
                    get_data_path(
                        dataset_name = 'artem/artem-tiny-imagenet',
                        local_root = os.path.expanduser('~/Documents/Scratch/tiny_imagenet/'),
                        local_repo = 'tiny-imagenet-200',
                        path = 'val/for_keras'
                    ),
                    "Path to store logs and checkpoints. It is recommended"
                    "to use get_logs_path() to define your logs directory."
                    "so that you can switch from local to clusterone without"
                    "changing your code."
                    "If you set your logs directory manually make sure"
                    "to use /logs/ when running on ClusterOne cloud.")
flags.DEFINE_string("log_dir",
                    get_logs_path(
                        os.path.expanduser('~/Documents/Scratch/tiny_imagenet/logs/')
                    ),
                    "Path to dataset. It is recommended to use get_data_path()"
                    "to define your data directory.so that you can switch "
                    "from local to clusterone without changing your code."
                    "If you set the data directory manually makue sure to use"
                    "/data/ as root path when running on ClusterOne cloud.")
FLAGS = flags.FLAGS

def device_and_target():
    # If FLAGS.job_name is not set, we're running single-machine TensorFlow.
    # Don't set a device.
    if FLAGS.job_name is None:
        print("Running single-machine training")
        return (None, "")

    # Otherwise we're running distributed TensorFlow
    print("Running distributed training")
    if FLAGS.task_index is None or FLAGS.task_index == "":
        raise ValueError("Must specify an explicit `task_index`")
    if FLAGS.ps_hosts is None or FLAGS.ps_hosts == "":
        raise ValueError("Must specify an explicit `ps_hosts`")
    if FLAGS.worker_hosts is None or FLAGS.worker_hosts == "":
        raise ValueError("Must specify an explicit `worker_hosts`")

    cluster_spec = tf.train.ClusterSpec({
        "ps": FLAGS.ps_hosts.split(","),
        "worker": FLAGS.worker_hosts.split(","),
    })
    server = tf.train.Server(
        cluster_spec, job_name=FLAGS.job_name, task_index=FLAGS.task_index)
    if FLAGS.job_name == "ps":
        server.join()

    worker_device = "/job:worker/task:{}".format(FLAGS.task_index)
    # The device setter will automatically place Variables ops on separate
    # parameter servers (ps). The non-Variable ops will be placed on the workers.
    return (
        tf.train.replica_device_setter(
        worker_device=worker_device,
        cluster=cluster_spec),
        server.target,
)

# --- end of snippet ----

#
# Data
#
def make_data_generators():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255.
    )
    generator_kwargs = {
        'target_size': (128, 128),
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'seed': SEED
    }

    data_gens = namedtuple('data_gens', ['train', 'val'])
    return data_gens(
        train=train_datagen.flow_from_directory(
            FLAGS.train_data_dir,
            **generator_kwargs
        ),
        val=val_datagen.flow_from_directory(
            FLAGS.val_data_dir,
            **generator_kwargs
        )
    )

#
# Model Definition
#
def make_model_def():
    # adopted from https://github.com/miquelmarti/tiny-imagenet-classifier/blob/master/tiny_imagenet_classifier.py

    model_inp = keras.layers.Input(shape=(128, 128, 3,))

    # conv-spatial batch norm - relu #1
    x = keras.layers.ZeroPadding2D((2, 2))(model_inp)
    x = keras.layers.Conv2D(64, (5, 5), strides=(2, 2),
                                   kernel_regularizer=keras.regularizers.l1_l2(l1=1e-7, l2=1e-7))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #2
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(64, (3, 3), strides=(1, 1))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #3
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(2, 2))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    # conv-spatial batch norm - relu #4
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(128, (3, 3), strides=(1, 1))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #5
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(2, 2))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #6
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(256, (3, 3), strides=(1, 1))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    # conv-spatial batch norm - relu #7
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(2, 2))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #8
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(512, (3, 3), strides=(1, 1))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)

    # conv-spatial batch norm - relu #9
    x = keras.layers.ZeroPadding2D((1, 1))(x)
    x = keras.layers.Conv2D(1024, (3, 3), strides=(2, 2))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.25)(x)

    # Affine-spatial batch norm -relu #10
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5))(x)
   # x = keras.layers.BatchNormalization(epsilon=1e-06, axis=-1, momentum=0.9)(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Dropout(0.5)(x)

    model_out = keras.layers.Dense(
        N_CLASSES, activation='softmax',
        kernel_regularizer=keras.regularizers.l1_l2(l1=1e-5, l2=1e-5)
    )(x)
    model = keras.models.Model(model_inp, model_out)

    return model

def main(_):
    data = make_data_generators()

    device, target = device_and_target()  # getting node environment
    with tf.device(device):
        # set Keras learning phase to train
        keras.backend.set_learning_phase(1)
        # do not initialize variables on the fly
        keras.backend.manual_variable_initialization(True)

        #
        # Keras model
        #
        model = make_model_def()

        # keras model predictions
        predictions = model.output
        # placeholder for training targets
        targets = tf.placeholder(tf.float32, shape=(None, N_CLASSES))
        # categorical crossentropy loss
        loss = tf.reduce_mean(
            keras.losses.categorical_crossentropy(targets, predictions)
        )
        # accuracy
        acc_value = tf.reduce_mean(keras.metrics.categorical_accuracy(targets, predictions))
        # global step
        global_step = tf.train.get_or_create_global_step()

        # Only if you have regularizers
        total_loss = loss * 1.0  # Copy
        for reg_loss in model.losses:
            total_loss = total_loss + reg_loss

        optimizer = tf.train.AdamOptimizer()

        # Barrier to compute gradients after updating moving avg of batch norm
        with tf.control_dependencies(model.updates):
            barrier = tf.no_op(name="update_barrier")

        with tf.control_dependencies([barrier]):
            grads = optimizer.compute_gradients(
                total_loss,
                model.trainable_weights
            )
            grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

        with tf.control_dependencies([grad_updates]):
            train_op = tf.identity(total_loss, name="train")

    #
    # Train
    #
    with tf.train.MonitoredTrainingSession(
            master=target, is_chief=(FLAGS.task_index == 0), checkpoint_dir=FLAGS.log_dir) as sess:

        for i in range(1000):
            batch_train_x, batch_train_y = data.train.next()

            # perform the operations defined earlier on batch
            loss_val = sess.run(
                [train_op],
                feed_dict={
                    model.inputs[0]: batch_train_x,
                    targets: batch_train_y
                }
            )
            # Add loss to tensorboard
            print('Batch Number: {0:4d}, Task: {1:3d}, Train Loss: {2:6.4f}'.format(i, FLAGS.task_index, loss_val[0]))

        #    if i % VAL_FREQUENCY_STEPS == 0:
        #        for batch_val_x, batch_val_y in data.val:
       #     val_acc = sess.run(
       #         acc_value,
       #         feed_dict={
       #             model.inputs[0]: mnist.test.images,
       #             targets: mnist.test.labels
       #         }
       #     )

       #     print('Batch Number: {0:4d}, Task: {1:3d}, Validation Accuracy: {2:6.4f}'.format(i, FLAGS.task_index, val_acc))

if __name__ == '__main__':
    tf.app.run()