import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values of the images from [0, 255] to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Reshape images to fit the CNN input structure
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Define the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5, validation_split=0.1)

import tkinter as tk
from tkinter import Canvas
import PIL.Image, PIL.ImageDraw

def save():
    # Save the current canvas to an image and predict the digit
    filename = "user_digit.jpg"
    image1.save(filename)
    predict_digit(filename)

def clear():
    # Clear the drawing canvas
    canvas.delete("all")
    draw.rectangle((0, 0, 200, 200), fill=(0, 0, 0, 0))

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)

app = tk.Tk()
canvas = Canvas(app, width=200, height=200, bg='white')
canvas.pack()
canvas.bind('<B1-Motion>', paint)

button_save = tk.Button(app, text="Save", command=save)
button_save.pack()
button_clear = tk.Button(app, text="Clear", command=clear)
button_clear.pack()

image1 = PIL.Image.new("RGB", (200, 200), (255, 255, 255))
draw = PIL.ImageDraw.Draw(image1)

app.mainloop()

def predict_digit(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.reshape(img_array, (1, 28, 28, 1)) / 255.0
    prediction = model.predict(img_array)
    predicted_digit = tf.argmax(prediction[0])
    print(f'Predicted Digit: {predicted_digit.numpy()}')
