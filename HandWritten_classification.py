import tensorflow as tf
class myCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs={}):
                if(logs.get('acc')>0.99):
                    self.model.stop_training = True
    

def train_mnist():
    
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.

    # YOUR CODE SHOULD START HERE
     
   
    
    # YOUR CODE SHOULD END HERE
    
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    import numpy as np
    np.set_printoptions(linewidth=200)
    plt.imshow(x_train[0])
    # YOUR CODE SHOULD START HERE
    x_train=x_train/255.0
    y_test=y_train/255.0
    callbacks = myCallback()

    # YOUR CODE SHOULD END HERE
    model = tf.keras.models.Sequential([
        # YOUR CODE SHOULD START HERE
        tf.keras.layers.Flatten(input_shape=(28,28)),
        tf.keras.layers.Dense(512,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation=tf.nn.softmax)
    
        # YOUR CODE SHOULD END HERE
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
   
    
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
              # YOUR CODE SHOULD END HERE
    
    # model fitting
    return history.epoch, history.history['acc'][-1]
