from pytube import YouTube
from youtube_transcript_api import YouTubeTranscriptApi as transcript
import re

def download_captions(url, path, title):
   code= url.split("v=")[-1] #get URL idcode for video
   title= re.sub('[^A-Za-z0-9 ]+', '', title)

   #get transcript
   srt= transcript.get_transcript(code, languages=['en'])
   
   #write captions to file, including starting time and duration
   with open(f'{path}/{title}.txt', "w") as f:
      for i in srt:
         f.write("{}\n".format(i))

         #general captions
      f.write("\n")
      for i in srt:
         f.write(f"{i['text']}\n")


def download_video(url, path='./Videos'):
   video= YouTube(url) #gets the url

   video_stream= video.streams.get_highest_resolution()
   video_stream.download(path) #downloads to file path
   
   download_captions(url, path, video.title)

#function to download video
url= ["https://www.youtube.com/watch?v=wbWRWeVe1XE",
      "https://www.youtube.com/watch?v=FlJoBhLnqko",
      "https://www.youtube.com/watch?v=Y-bVwPRy_no"]

for i in url:
   download_video(i)