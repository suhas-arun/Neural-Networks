"""
Handwritten digit classifier using a feedforward neural network, trained
using the MNIST dataset.
"""
from tensorflow import keras

# Load the MNIST dataset
MNIST = keras.datasets.mnist

(train_images, train_labels), (test_images, test_labels) = MNIST.load_data()

model = keras.Sequential(
    [
        keras.layers.Flatten(input_shape=(28, 28)),
        # the sigmoid activation function is used
        keras.layers.Dense(128, activation="sigmoid"),
        keras.layers.Dense(10),
    ]
)
model.compile(
    # Stochastic gradient descent is used and the mean squared error is the cost function
    optimizer="SGD",
    loss=keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

model.fit(train_images, keras.utils.to_categorical(train_labels), epochs=10)

test_loss, test_acc = model.evaluate(
    test_images, keras.utils.to_categorical(test_labels), verbose=2
)
