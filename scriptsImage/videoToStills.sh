#!/bin/bash
if [ -z "$1" ]
then
      echo "Usage: videoToStills.sh video outputDir seconds"
      exit
fi
if [ -z "$2" ]
then
      echo "Usage: videoToStills.sh video outputDir seconds"
      exit
fi
if [ -z "$3" ]
then
      echo "Usage: videoToStills.sh video outputDir seconds"
      exit
fi

#time for i in {0..$3} ; do ffmpeg -accurate_seek -ss `echo $i*1.0 | bc` -i "$1" -frames:v 1 "$2/frame_$i.bmp" ; done
START=0
for (( i=$START; i<=$3; i++ ))
do
    ffmpeg -accurate_seek -ss `echo $i*1.0 | bc` -i "$1" -frames:v 1 "$2/frame_$i.bmp"
done
