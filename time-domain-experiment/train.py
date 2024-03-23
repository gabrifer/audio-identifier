from pyexpat.model import XML_CQUANT_REP
import numpy as np
import json
from sklearn import svm
from sklearn.model_selection import train_test_split

DATA_PATH = "dataset.json"

def load_dataset(data_path):
    with open(data_path,"r") as fp:
        data = json.load(fp)
       
    X = np.array(data["features"])
    y = np.array(data["labels"])
    
    return X, y

def get_data_splits(data_path, test_size=0.2):
    
    X, y = load_dataset(data_path)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = get_data_splits(DATA_PATH)
    
    model = svm.SVC(verbose=True)
    
    model.fit(X_train, y_train)
    
    y_result = model.predict(X_train)
    
    for i, _ in enumerate(y_result):
        print(f'Expected: {y_test[i]} | Predicted {y_result[i]}')

main()