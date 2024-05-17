import json
import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import *
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

DATA_PATH = "dataset.json"
LEARNING_RATE = 0.0001
EPOCHS = 40
BATCH_SIZE = 32

def load_dataset(data_path):
    
    with open(data_path,"r") as fp:
       data = json.load(fp)
       
    X = np.array(data["joined_time_domain_features"])
    y = np.array(data["labels"])
    
    return X, y

def main():
    # Load dataset
    X, y = load_dataset(DATA_PATH)
    
    # Split between training and testing dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Standardizing the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    # Creating an SVM classifier
    clf = svm.SVC(kernel='rbf')  # You can change the kernel type as needed

    # Performing 10-fold cross-validation
    scores = cross_val_score(clf, X_scaled, y, cv=10)

    # Printing the accuracy for each fold
    print("Accuracy for each fold:", scores)
    
    y_pred = cross_val_predict(clf, X_scaled, y, cv=10)

    # Calculating and printing the mean accuracy
    mean_accuracy = np.mean(scores)
    print("Mean Accuracy:", mean_accuracy)
    
    # Getting the confusion matrix
    conf_matrix = confusion_matrix(y, y_pred)
    print("Confusion Matrix:")
    print(conf_matrix)
    
    print(f"Recall score: {recall_score(y, y_pred)}")
    print(f"Precision score: {precision_score(y, y_pred)}")
    print(classification_report(y,y_pred))


main()