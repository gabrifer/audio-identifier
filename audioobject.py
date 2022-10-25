import librosa
import constants
import featureextractor

class Audio:
    
    def __init__(self, audio_id):
        self.audio_id = audio_id
        self.audio_signal, self.audio_sr = librosa.load("splitted-audio/" + self.audio_id)
        self.audio_spectrogram = librosa.stft(self.audio_signal, n_fft=constants.FRAME_LENGTH, hop_length=constants.HOP_LENGTH)
        self.audio_features = {}
        self.audio_label = ""
        
    def __str__(self):
        output = ""
        for k, v in self.audio_features.items():
             output = output + str(k) + " " + str(v) + " \n"
        return output
        
    def extract_features(self):
        print("Extracting audio features for audio ID " + self.audio_id)
        self.audio_features[constants.AMPLITUDE_ENVELOPE] = \
            featureextractor.get_amplitude_envelope(self.audio_signal, constants.FRAME_LENGTH, constants.HOP_LENGTH)
        self.audio_features[constants.RMS_ENERGY] = \
            featureextractor.get_rms_energy(self.audio_signal, constants.FRAME_LENGTH, constants.HOP_LENGTH)
        self.audio_features[constants.ZERO_CROSSING_RATE] = \
            featureextractor.get_zero_crossing_rate(self.audio_signal, constants.FRAME_LENGTH, constants.HOP_LENGTH)
        self.audio_features[constants.MFCCS] = \
            featureextractor.get_mfccs(self.audio_signal, constants.N_MFCCS, self.audio_sr)
        self.audio_features[constants.BAND_ENERGY_RATIO] = \
            featureextractor.get_band_energy_ratio(self.audio_spectrogram, constants.SPLIT_FREQUENCY, self.audio_sr)
        
    