import json
import os
import librosa
import warnings

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

DATASET_PATH = '../splitted-audio-annotated'
JSON_PATH = 'dataset.json'
SAMPLES_TO_CONSIDER = 22050 * 5

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    
    print("Preparing dataset")
    
    # dictionary
    data = {
        "mappings": [],
        "labels": [],
        "MFCCs": [],
        "files": []
    }
    
    for i, (dirpath, _, filenames) in enumerate(os.walk(dataset_path)):
        
        if dirpath is not dataset_path:
            
            label = dirpath.split("/")[2]
            data["mappings"].append(label)
            
            print("Processing label: " + label)
            
            for f in filenames:
                
                print("Processing file: " + f)
                file_path = os.path.join(dirpath, f)
                
                signal, sr = librosa.load(file_path)
                
                if len(signal) < SAMPLES_TO_CONSIDER:
                    continue
                else:
                    signal = signal[:SAMPLES_TO_CONSIDER]
                    
                    mfccs = librosa.feature.mfcc(y=signal, n_mfcc=n_mfcc, hop_length=hop_length, n_fft=n_fft)
                    
                    data["labels"].append(i-1)
                    
                    data["MFCCs"].append(mfccs.tolist())
                    
                    data["files"].append(f)
                    
    
    print(f"Processed {len(data['files'])} files")                
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
                
                
prepare_dataset(DATASET_PATH, JSON_PATH)