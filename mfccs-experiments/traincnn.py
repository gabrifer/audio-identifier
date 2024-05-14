import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, classification_report


DATA_PATH = "dataset.json"
LEARNING_RATE = 0.0001
EPOCHS = 30
BATCH_SIZE = 32

def load_dataset(data_path):
    
    with open(data_path,"r") as fp:
       data = json.load(fp)
       
    X = np.array(data["MFCCs"])
    y = np.array(data["labels"])
    
    return X, y

def get_data_splits(data_path, test_size=0.15, validation_size=0.15):
    
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
    model.add(tf.keras.layers.Dense(2, activation='softmax'))

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
    # x_train, x_validation, x_test, y_train, y_validation, y_test = get_data_splits(DATA_PATH)
    
    # Load dataset
    X, y = load_dataset(DATA_PATH)
    
    # Get training dataset and test dataset
    X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size=0.2)
    
    # Define 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True)

    # Initialize lists to store accuracy for each fold
    acc_per_fold = []
    
    # Take the best model in terms of accuracy
    best_model = None
    
    # Current best accuracy
    best_accuracy = 0
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"Training Fold {fold_idx+1}...")
    
        # Split data into training and validation sets
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        X_train_fold = X_train_fold[..., np.newaxis]
        X_val_fold = X_val_fold[..., np.newaxis]
        
        # Create CNN model
        input_shape = (X_train_fold.shape[1], X_train_fold.shape[2], X_train_fold.shape[3])
        model = build_model(input_shape, LEARNING_RATE)
        
        
        # Train the model
        model.fit(X_train_fold, y_train_fold, epochs=EPOCHS, batch_size=32)
        
        # Evaluate the model on validation data
        scores = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        print(model.metrics_names)
        print(scores)
        acc_per_fold.append(scores[1])
        
        if scores[1] > best_accuracy:
            best_model = model
    
        print(f"Validation Accuracy for Fold {fold_idx+1}: {scores[1]}")

    # Calculate and print the average accuracy across all folds
    print(f"\nAverage Validation Accuracy: {np.mean(acc_per_fold)}")
    
    
    X = X[..., np.newaxis]
    
    y_pred = best_model.predict(X)
    y_pred_normalized = np.zeros(shape=(len(y_pred)))
    
    for idx, pred in enumerate(y_pred):
        if pred[0] > pred [1]:
            y_pred_normalized[idx] = 0
        else:
            y_pred_normalized[idx] = 1
             
    print(y)
    print(y_pred_normalized)
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(y, y_pred_normalized)

    print("Confusion Matrix:")
    print(conf_matrix)
    
    print(classification_report(y,y_pred_normalized))
    
    
    # train the model
    # model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(x_validation, y_validation))
    
    # test_loss, test_acc = model.evaluate(x_test, y_test)
    # print("\nTest loss: {}, test accuracy: {}".format(test_loss, 100*test_acc))
    

main()