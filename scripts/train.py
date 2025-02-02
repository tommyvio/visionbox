import tensorflow as tf
from models.cnn_model import build_cnn_model
from utils.data_loader import load_data
from utils.visualization import plot_training_history

def train_model():
    # Load data
    (train_images, train_labels), (test_images, test_labels) = load_data()

    # Preprocess data
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    # Build model
    model = build_cnn_model(input_shape=train_images.shape[1:], num_classes=10)

    # Compile model
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Train model
    history = model.fit(train_images, train_labels, epochs=10,
                        validation_data=(test_images, test_labels))

    # Save model
    model.save('models/visionbox_cnn.h5')

    # Plot training history
    plot_training_history(history)

if __name__ == "__main__":
    train_model()
