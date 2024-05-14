from pydub import AudioSegment
import os
import fileutils

def get_audio_chunk_name(audio_file_name, chunk_number):
    return audio_file_name[:-4] + "-chunk" + str(chunk_number)

def split_audios():

    print("Starting audios splitting")

    extracted_audios_path = os.getcwd() + "/extracted-audio/"
    splitted_audios_path =  os.getcwd() + "/splitted-audio"

    extracted_audios_files_list = os.listdir(extracted_audios_path)

    for audio_file in extracted_audios_files_list:

        audio_path = extracted_audios_path + audio_file

        # Check if this audio file was already splitted
        if fileutils.find_file_by_name(get_audio_chunk_name(audio_file, 0) + ".mp3", splitted_audios_path) != None:
            print("Skipping audio splitting since it was already splitted: " + audio_file)
            continue

        audio = AudioSegment.from_file(audio_path)

        splitted_audios = audio[::5000]

        for i, splitted_audio in enumerate(splitted_audios):
            splitted_audio.export(os.getcwd() + "/splitted-audio/" + get_audio_chunk_name(audio_file, i) + ".mp3", format="mp3")
            print("Audio " + audio_file + " splitted in chunk #" + str(i))
        
    print("Audio splitting completed")
