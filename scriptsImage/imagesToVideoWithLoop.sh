#!/bin/sh
if [ -z "$1" ]
then
      echo "Usage: imagesToVideoWithLoop.sh inputDir ouputFile"
      exit
fi
if [ -z "$2" ]
then
      echo "Usage: imagesToVideoWithLoop.sh inputDir ouputFile"
      exit
fi

ffmpeg -i $1/%5d.png -c:v libx264 -vf fps=25 -pix_fmt yuv420p -b:v 3000k $2
NEWNAME=${2%%.mp4}.loop.mp4
ffmpeg -i $2 -filter_complex "[0:v]reverse,fifo[r];[0:v][r] concat=n=2:v=1 [v]" -map "[v]" $NEWNAME
mv $NEWNAME $2
#ffmpeg -i generateResult/%5d.png -c:v libx264 -vf fps=5 -pix_fmt yuv420p -b:v 3000k out.mp4
