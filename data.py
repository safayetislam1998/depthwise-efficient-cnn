import tensorflow as tf
from augment import RandAugment, AutoAugment, MixupAndCutmix

AUTO = tf.data.AUTOTUNE

def preprocess_image(image, label):
    image = tf.cast(image, dtype=tf.float32) / 255.0
    label = tf.cast(label, dtype=tf.float32)
    return image, label

def prepare_dataset(batch_size=512, num_classes=10):
    # Load CIFAR-10
    (x_train_root, y_train_root), (x_test_root, y_test_root) = tf.keras.datasets.cifar10.load_data()
    y_train_onehot = tf.keras.utils.to_categorical(y_train_root, num_classes=num_classes)
    y_test_onehot = tf.keras.utils.to_categorical(y_test_root, num_classes=num_classes)
    

    mixup_cutmix = MixupAndCutmix(num_classes=num_classes)
    rand_aug = RandAugment(num_layers=2, magnitude=10.)
    auto_aug = AutoAugment()
    
    # Create testing dataset
    test_ds = tf.data.Dataset.from_tensor_slices((x_test_root, y_test_onehot))
    test_ds = (test_ds
               .map(preprocess_image, num_parallel_calls=AUTO)
               .batch(batch_size)
               .prefetch(AUTO))
    
    # Create training dataset with augmentations
    train_ds = tf.data.Dataset.from_tensor_slices((x_train_root, y_train_onehot))
    train_ds = (train_ds
                .repeat(1000) 
                .shuffle(50000, reshuffle_each_iteration=True)
                .batch(batch_size)
                .map(mixup_cutmix, num_parallel_calls=AUTO)  # Apply mixup/cutmix
                .map(lambda x, y: (rand_aug.distort(x), y), num_parallel_calls=AUTO)  # Random augment
                .map(lambda x, y: (auto_aug.distort(x), y), num_parallel_calls=AUTO)  # Auto augment
                .map(preprocess_image, num_parallel_calls=AUTO)
                .prefetch(AUTO))
    
    return train_ds, test_ds, len(x_train_root)