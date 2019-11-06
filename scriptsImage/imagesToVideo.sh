#!/bin/sh
if [ -z "$1" ]
then
      echo "Usage: imagesToVideo.sh inputDir ouputFile"
      exit
fi
if [ -z "$2" ]
then
      echo "Usage: imagesToVideo.sh inputDir ouputFile"
      exit
fi

ffmpeg -i $1/%5d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p -b:v 3000k $2
#ffmpeg -i generateResult/%5d.png -c:v libx264 -vf fps=5 -pix_fmt yuv420p -b:v 3000k out.mp4
