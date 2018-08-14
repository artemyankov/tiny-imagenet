from keras.preprocessing.image import ImageDataGenerator
from keras.applications import MobileNetV2
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.optimizers import Adam
from clusterone import get_data_path, get_logs_path

N_CLASSES = 200
BATCH_SIZE = 32

train_data_dir = get_data_path(
    dataset_name = 'artem/artem-tiny-imagenet',
    local_root = '/Users/artem/Documents/Scratch/tiny_imagenet/',
    local_repo = 'tiny-imagenet-200',
    path = 'train'
)

val_data_dir = get_data_path(
    dataset_name = 'artem/artem-tiny-imagenet',
    local_root = '/Users/artem/Documents/Scratch/tiny_imagenet/',
    local_repo = 'tiny-imagenet-200',
    path = 'val/for_keras'
)

models_dir = get_data_path(
    dataset_name = 'artem/artem-tiny-imagenet',
    local_root = '/Users/artem/Documents/Scratch/tiny_imagenet/',
    local_repo = '',
    path = 'models'
)

log_dir = get_logs_path('/Users/artem/Documents/Scratch/tiny_imagenet/logs/')

def train():

    #
    # Data Preparation
    #

    train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True
    )
    val_datagen = ImageDataGenerator(
        rescale = 1. / 255.
    )

    generator_kwargs = {
        'target_size': (224, 224),
        'batch_size': BATCH_SIZE,
        'shuffle': True,
        'seed': 43543
    }

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        **generator_kwargs
    )
    val_generator = val_datagen.flow_from_directory(
        val_data_dir,
        **generator_kwargs
    )

    #
    # Model
    #

    model = MobileNetV2(
        include_top = True,
        weights = None,
        classes = N_CLASSES
    )
    model.compile(
        loss = 'categorical_crossentropy',
        optimizer = Adam(),
        metrics = ['accuracy']
    )

    #
    # Checkpoints
    #

    checkpoint = ModelCheckpoint(
        filepath = os.path.join(models_dir, 'weights.{epoch:02d}-{val_loss:.2f}.h5'),
        monitor = 'val_acc',
        verbose = 1,
        save_best_only = True
    )

    tensorboard = TensorBoard(log_dir = log_dir)

    early_stopping = EarlyStopping(
        monitor = 'val_loss',
        patience = 2
    )

    callbacks = [checkpoint, tensorboard, early_stopping]

    #
    # Train
    #

    model.fit_generator(
        train_generator,
        epochs = 1,
        validation_data = val_generator,
        callbacks = callbacks
    )

if __name__ == '__main__':
    train()