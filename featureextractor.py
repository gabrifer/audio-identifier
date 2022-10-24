import math
from pickle import FRAME
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt
import numpy as np

FRAME_LENGTH = 1024
HOP_LENGTH = 1024
N_MFCCS = 13
SPLIT_FREQUENCY = 2000

def get_amplitude_envelope(signal, frame_length, hop_length):
    return np.array([max(signal[i:i+frame_length]) for i in range(0, signal.size, hop_length)])

def get_rms_energy(signal, frame_length, hop_length):
    return librosa.feature.rms(y=signal, frame_length=frame_length,hop_length=hop_length)

def get_zero_crossing_rate(signal, frame_length, hop_length):
    return librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_length, hop_length=hop_length)

def get_mfccs(signal, n_mfcc, sr):
    return librosa.feature.mfcc(signal, sr=sr, n_mfcc=n_mfcc)

def get_band_energy_ratio(spectrogram, split_frequency, sr):
    # First, calculate the split frequency bin
    frequency_range = sr / 2
    frequency_delta_per_bin = frequency_range / spectrogram.shape[0]
    split_frequency_bin = int(math.floor(split_frequency / frequency_delta_per_bin))
    
    band_energy_ratio = []
    
    # calculate power spectrogram
    power_spectrogram = np.abs(spectrogram) ** 2
    power_spectrogram = power_spectrogram.T
    
    # calculate BER value for each frame
    for frame in power_spectrogram:
        sum_power_low_frequencies = frame[:split_frequency_bin].sum()
        sum_power_high_frequencies = frame[split_frequency_bin:].sum()
        band_energy_ratio_current_frame = sum_power_low_frequencies / sum_power_high_frequencies
        band_energy_ratio.append(band_energy_ratio_current_frame)
    
    return np.array(band_energy_ratio) 
    

def plot_signal(signal, title):
    plt.figure(figsize=(16,17))
    plt.subplot(1,1,1)
    ld.waveshow(signal)
    plt.title(title)
    plt.ylim((-1,1))

    plt.show()
    

signal, sr = librosa.load('splitted-audio/_79oAUT37Og-chunk1.mp3')
spectrogram = librosa.stft(signal, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)

print(get_amplitude_envelope(signal, FRAME_LENGTH, HOP_LENGTH))
print(get_rms_energy(signal, FRAME_LENGTH, HOP_LENGTH))
print(get_zero_crossing_rate(signal, FRAME_LENGTH, HOP_LENGTH))
print(get_mfccs(signal, N_MFCCS, sr))
print(get_band_energy_ratio(spectrogram, SPLIT_FREQUENCY, sr))




