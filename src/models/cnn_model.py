"""Defines a basic CNN model for audio classification."""

import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    """Builds a simple CNN model.

    Args:
        input_shape: The shape of the input features 
                     (e.g., (num_frames, num_mfcc_coeffs, num_channels)).
                     Adjust based on how features are preprocessed.
        num_classes: The number of output classes (e.g., 2 for drone/no-drone).

    Returns:
        A compiled Keras Sequential model.
    """
    model = models.Sequential(name="SimpleCNN")

    # Convolutional layers
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten and Dense layers
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5)) # Dropout for regularization

    # Output layer
    if num_classes == 2:
        # Binary classification
        model.add(layers.Dense(1, activation='sigmoid')) 
        loss = 'binary_crossentropy'
    else:
        # Multi-class classification
        model.add(layers.Dense(num_classes, activation='softmax'))
        loss = 'sparse_categorical_crossentropy' # Use if labels are integers
        # loss = 'categorical_crossentropy' # Use if labels are one-hot encoded

    # Compile the model
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) # Example optimizer
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'] # Add other relevant metrics if needed
    )

    model.summary() # Print model summary
    return model

if __name__ == "__main__":
    # Example usage:
    # Define dummy input shape and number of classes
    # Adjust these based on your actual feature dimensions
    dummy_input_shape = (44, 40, 3) # Example: (time_frames, mfcc_coeffs, channels=stacked features)
    dummy_num_classes = 2 

    print(f"Building CNN model with input shape: {dummy_input_shape} and {dummy_num_classes} classes...")
    cnn_model = build_cnn_model(dummy_input_shape, dummy_num_classes)
    print("CNN model built successfully.")
    # You can inspect the model further here if needed
    # print(cnn_model.layers) 