#!/bin/bash
if [ -z "$1" ]
then
      echo "Usage: fasterVideo.sh videoPath"
      exit
fi

NEWNAME=${1%%.mp4}.faster.mp4

ffmpeg -i $1 -filter:v "setpts=0.5*PTS" $NEWNAME
vlc $NEWNAME

