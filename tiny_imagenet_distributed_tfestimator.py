import tensorflow as tf
import os
import json
from clusterone import get_data_path, get_logs_path
from model_def import make_model_def
import tiny_imagenet_data_utils as data_utils

flags = tf.app.flags

#
# Snippet for distributed learning
#
try:
    config = os.environ['TF_CONFIG']
    config = json.loads(config)
    task = config['task']['type']
    task_index = config['task']['index']

    local_ip = 'localhost:' + config['cluster'][task][task_index].split(':')[1]
    config['cluster'][task][task_index] = local_ip
    if task == 'chief' or task == 'master':
        config['cluster']['worker'][task_index] = local_ip
    os.environ['TF_CONFIG'] = json.dumps(config)
except:
    pass
#
#
#

# Training related flags
flags.DEFINE_integer("N_CLASSES",
                     200,
                     "Number of prediction classes in the model")
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

def main(_):
    train_input = lambda: data_utils.input_fn(
        FLAGS.train_data_dir,
        num_epochs=1,
        batch_size=32
    )

    val_input = lambda: data_utils.input_fn(
        FLAGS.val_data_dir,
        shuffle=False
    )

    model = make_model_def()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.categorical_crossentropy,
        metrics=['accuracy']
    )

    config = tf.estimator.RunConfig(
        model_dir=FLAGS.log_dir,
        save_summary_steps=500,
        save_checkpoints_steps=500,
        keep_checkpoint_max=5,
        log_step_count_steps=50
    )

    classifier = tf.keras.estimator.model_to_estimator(
        model,
        model_dir=FLAGS.log_dir,
        config=config
    )

    train_spec = tf.estimator.TrainSpec(
        input_fn=train_input,
        max_steps=1e6
    )
    val_spec = tf.estimator.EvalSpec(
        input_fn=val_input,
        steps=None,
        start_delay_secs=0,
        throttle_secs=1
    )

    tf.estimator.train_and_evaluate(classifier, train_spec, val_spec)

if __name__ == '__main__':
    tf.app.run()
