import os
import librosa
import numpy
import json
import warnings

warnings.filterwarnings(
    "ignore", message="PySoundFile failed. Trying audioread instead.")

DATASET_PATH = '../splitted-audio-annotated'
JSON_PATH = 'dataset.json'
FRAME_SIZE = 1024
HOP_LENGTH = 512


def get_amplitude_envelope(signal, frame_size, hop_length):

    amplitude_envelope = []

    for i in range(0, len(signal), hop_length):
        amplitude_envelope.append(str(max(signal[i:i+frame_size])))

    return amplitude_envelope


def get_rms_energy(signal, frame_size, hop_length):

    rms = []

    for i in range(0, len(signal), hop_length):
        rms_current_frame = numpy.sqrt(
            numpy.sum(signal[i:i+frame_size]**2) / frame_size)
        rms.append(str(rms_current_frame))

    return rms


def prepare_dataset(dataset_path, json_path):

    print("Preparing dataset for using time domain features")

    # dictionary
    data = {
        "mappings": [],
        "labels": [],
        "zcr": [],
        "amplitude_envelope": [],
        "rms": [],
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

                # Extract Time Domain features

                zcr = librosa.feature.zero_crossing_rate(y=signal)

                amplitude_envelope = get_amplitude_envelope(
                    signal, FRAME_SIZE, HOP_LENGTH)

                rms = get_rms_energy(signal, FRAME_SIZE, HOP_LENGTH)

                data["labels"].append(i-1)

                data["zcr"].append([zcr.min(), zcr.max(), zcr.mean()])

                data["amplitude_envelope"].append(amplitude_envelope)

                data["rms"].append(rms)

                data["files"].append(f)

    with open(json_path, "w") as fp:
        json.dump(data, fp, indent=4)


prepare_dataset(DATASET_PATH, JSON_PATH)
