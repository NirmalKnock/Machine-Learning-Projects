
#Importing tensorflow
import tensorflow as tf
print(tf.__version__)

#Load data from MNIST
mnist = tf.keras.datasets.mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()
training_images=training_images/255.0
test_images=test_images/255.0

#Tranning a modeil
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Optomizing

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
model.fit(training_images, training_labels, epochs=5)

#Evaluvation
model.evaluate(test_images, test_labels)

#Predict

classifications = model.predict(test_images)
print(classifications[0])
print(test_labels[0])
