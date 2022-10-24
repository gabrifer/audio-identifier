from pickle import FRAME
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np

FRAME_LENGTH = 1024
HOP_LENGTH = 1024

def get_amplitude_envelope(signal, frame_length, hop_length):
    return np.array([max(signal[i:i+frame_length]) for i in range(0, signal.size, hop_length)])

def get_rms_energy(signal, frame_length, hop_length):
    return librosa.feature.rms(y=signal, frame_length=frame_length,hop_length=hop_length)

def get_zero_crossing_rate(signal, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)

def plot_signal(signal, title):
    plt.figure(figsize=(16,17))
    plt.subplot(1,1,1)
    ld.waveshow(signal)
    plt.title(title)
    plt.ylim((-1,1))

    plt.show()
    

signal, sr = librosa.load('splitted-audio/_79oAUT37Og-chunk1.mp3')

print(get_amplitude_envelope(signal, FRAME_LENGTH, HOP_LENGTH))
print(get_rms_energy(signal, FRAME_LENGTH, HOP_LENGTH))
print(get_zero_crossing_rate(signal, FRAME_LENGTH, HOP_LENGTH))




