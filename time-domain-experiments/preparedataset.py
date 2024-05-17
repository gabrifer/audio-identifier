import json
import os
import librosa
import warnings
import numpy as np

warnings.filterwarnings("ignore", message="PySoundFile failed. Trying audioread instead.")

DATASET_PATH = '../splitted-audio-annotated'
JSON_PATH = 'dataset.json'
SAMPLES_TO_CONSIDER = 22050 * 5
FRAME_LENGTH = 1024
HOP_LENGTH = 1024

def get_amplitude_envelope(signal, frame_length, hop_length):
    return np.array([max(signal[i:i+frame_length]) for i in range(0, signal.size, hop_length)])

def get_rms_energy(signal, frame_length, hop_length):
    return librosa.feature.rms(y=signal, frame_length=frame_length,hop_length=hop_length)

def get_zero_crossing_rate(signal, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)

def prepare_dataset(dataset_path, json_path, n_mfcc=13, hop_length=512, n_fft=2048):
    
    print("Preparing dataset")
    
    # dictionary
    data = {
        "mappings": [],
        "labels": [],
        "amplitude_envelope": [],
        "rms_energy": [],
        "zero_crossing_rate": [],
        "joined_time_domain_features": [],
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
                    
                    ae = get_amplitude_envelope(signal, FRAME_LENGTH, HOP_LENGTH)
                    
                    rms = get_rms_energy(signal, FRAME_LENGTH, HOP_LENGTH)
                    
                    zcr = get_zero_crossing_rate(signal, FRAME_LENGTH, HOP_LENGTH)
                    
                    data["labels"].append(i-1)
                    
                    data["amplitude_envelope"].append(ae.tolist())
                    
                    data["rms_energy"].append(rms.tolist())
                    
                    data["zero_crossing_rate"].append(zcr.tolist())
                    
                    data["joined_time_domain_features"].append(np.concatenate((ae, rms[0], zcr[0])).tolist())
                    
                    data["files"].append(f)
                    
    
    print(f"Processed {len(data['files'])} files")                
                    
    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)
                
                
prepare_dataset(DATASET_PATH, JSON_PATH)