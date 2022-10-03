import os
import re

from pytube import YouTube

def find_file_by_name(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

destination = "/home/gabriel/masters-code/extrated-audio"

pattern = r'(?:https?:\/\/)?(?:[0-9A-Z-]+\.)?(?:youtube|youtu|youtube-nocookie)\.(?:com|be)\/(?:watch\?v=|watch\?.+&v=|embed\/|v\/|.+\?v=)?([^&=\n%\?]{11})'

# Loading video URLs from external file
video_list_file_name = "video-list.txt"
video_list = open(video_list_file_name, "r")
video_urls = video_list.read()
video_urls_list = video_urls.split("\n")
video_list.close()

for video_url in video_urls_list:

    # Extract video ID from YouTube URL
    video_id = re.search(pattern,  video_url).groups()[0]
    file_name = video_id + ".mp3"

    # Check if video was already downloaded
    if find_file_by_name(file_name, destination) != None:
        print("Skipping video because it was already downloaded: " + file_name)
        continue

    video = YouTube(video_url)
    audio = video.streams.filter(only_audio=True, file_extension='mp4').first()

    audio.download(destination, file_name)

    print("Completed download of video ID:" + video_id)
