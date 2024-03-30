#Problem 1- Video Library
from pytube import YouTube
import re


def download_video(url, path='./videos'):
   video= YouTube(url) #gets the url
   title= re.sub('[^A-Za-z0-9]+', '', video.title)

   video_stream= video.streams.get_highest_resolution()
   video_stream.download(path, filename= f'{title}.mp4') #downloads to file path

#function to download video
url= ["https://youtu.be/WeF4wpw7w9k",
      "https://youtu.be/2NFwY15tRtA",
      "https://youtu.be/5dRramZVu2Q"]

for i in url:
   download_video(i)