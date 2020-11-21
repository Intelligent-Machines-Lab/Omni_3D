import os

os.system("ffmpeg -r 5 -i animations/global-%01d.png -vcodec mpeg4 -y -vb 100M animations/global.mp4")
os.system("ffmpeg -r 5 -i animations/feature-%01d.png -vcodec mpeg4 -y -vb 100M animations/feature.mp4")
os.system("ffmpeg -r 5 -i animations/original-%01d.png -vcodec mpeg4 -y -vb 100M animations/original.mp4")