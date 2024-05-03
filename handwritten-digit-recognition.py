import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tkinter as tk
from tkinter import Canvas
import PIL.Image, PIL.ImageDraw
import ssl
import certifi
import urllib.request

def predict_digit(filename):
    img = tf.keras.preprocessing.image.load_img(filename, target_size=(28, 28), color_mode="grayscale")
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.reshape(img_array, (1, 28, 28, 1)) / 255.0
    prediction = model.predict(img_array)
    predicted_digit = tf.argmax(prediction[0])
    print(f'Predicted Digit: {predicted_digit.numpy()}')

# Create a SSL context object with certificates from certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# Use this context when fetching URLs
response = urllib.request.urlopen("https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz", context=ssl_context)

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the pixel values of the images from [0, 255] to [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# Define the CNN model with added dropout layers
model = models.Sequential([
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model using the data generator
model.fit(datagen.flow(train_images, train_labels, batch_size=32),
          epochs=50,
          validation_data=(test_images, test_labels),
          callbacks=[early_stopping])

# UI setup with Tkinter
def save():
    filename = "user_digit.jpg"
    image1.save(filename)
    predict_digit(filename)

def clear():
    canvas.delete("all")
    draw.rectangle((0, 0, 200, 200), fill=(0, 0, 0, 0))

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)

app = tk.Tk()
canvas = Canvas(app, width=800, height=800, bg='white')
canvas.pack()
canvas.bind('<B1-Motion>', paint)

button_save = tk.Button(app, text="Save", command=save)
button_save.pack()
button_clear = tk.Button(app, text="Clear", command=clear)
button_clear.pack()

image1 = PIL.Image.new("RGB", (200, 200), (255, 255, 255))
draw = PIL.ImageDraw.Draw(image1)

app.mainloop()
