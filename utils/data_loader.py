import tensorflow as tf

def load_data():
    # Example: Load CIFAR-10 dataset
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    return (train_images, train_labels), (test_images, test_labels)
