import os

from audioobject import Audio
# Read all audio files from folders
splitted_audios_path = os.getcwd() + "/splitted-audio"
extracted_audios_files_list = sorted(os.listdir(splitted_audios_path))

# Extract features
audio_objects = []
for audio_file in extracted_audios_files_list:
    new_audio_file = Audio(audio_file)
    new_audio_file.extract_features()
    audio_objects.append(new_audio_file)


# Store on DB