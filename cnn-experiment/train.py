import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = "dataset.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

def load_dataset(data_path):
    
    with open(data_path,"r") as fp:
       data = json.load(fp)
       
    x = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return x, y

def get_data_splits(data_path, test_size=0.2, validation_size=0.2):
    
   x, y = load_dataset(data_path)
   
   x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=test_size)
   x_train, x_validation, y_train, y_validation =  train_test_split(x, y, test_size=validation_size)
   
   x_train = x_train[..., np.newaxis]
   x_test = x_test[..., np.newaxis]
   x_validation = x_validation[..., np.newaxis]
   
   return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape, learning_rate, loss="sparse_categorical_crossentropy"):
    # build network architecture using convolutional layers
    model = tf.keras.models.Sequential()

    # 1st conv layer
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape,
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2,2), padding='same'))

    # 3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu',
                                     kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2,2), padding='same'))

    # flatten output and feed into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    tf.keras.layers.Dropout(0.3)

    # softmax output layer
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    optimiser = tf.optimizers.Adam(learning_rate=learning_rate)

    # compile model
    model.compile(optimizer=optimiser,
                  loss=loss,
                  metrics=["accuracy"])

    # print model parameters on console
    model.summary()

    return model
    

def main():
    
    # load train, test and validation data
    x_train, x_validation, x_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
    
    # build CNN model
    input_shape = (x_train.shape[1], x_train.shape[2], x_train.shape[3])
    model = build_model(input_shape, LEARNING_RATE)
    
    # train the model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_validation, y_validation))
    
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    

main()