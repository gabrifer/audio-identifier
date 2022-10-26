import os
import sys
import pickledb
import jsonpickle
from audioobject import Audio

db = pickledb.load("audio-features.db", True)

# Read all audio files from folders
splitted_audios_path = os.getcwd() + "/splitted-audio"
extracted_audios_files_list = sorted(os.listdir(splitted_audios_path))

# Filter out audio where features were already extracted
filtered_audio_list = list(filter(lambda audio_id : not db.exists(audio_id), extracted_audios_files_list))


if len(filtered_audio_list) == 0:
    print("Exiting script since all audio features are already stored")
    sys.exit()

# Extract features
audio_objects = []
for audio_file in filtered_audio_list:
    new_audio_file = Audio(audio_file)
    new_audio_file.extract_features()
    audio_objects.append(new_audio_file)


# Store on DB
print("Storing in local DB")
for audio in audio_objects:
    audio_json = jsonpickle.encode(audio, unpicklable=False)
    db.set(audio.audio_id, audio_json)