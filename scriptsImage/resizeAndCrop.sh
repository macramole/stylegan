#!/bin/bash
if [ -z "$1" ]
then
      echo "Usage: resizeAndCrop.sh width height imageDir"
      exit
fi
if [ -z "$2" ]
then
      echo "Usage: resizeAndCrop.sh width height imageDir"
      exit
fi
if [ -z "$3" ]
then
      echo "Usage: resizeAndCrop.sh width height imageDir"
      exit
fi
width=$1
height=$2
images=$3

if [ ! -d "$images" ]; then
    echo "Usage: resizeAndCrop.sh width height imageDir"
    echo ""
    echo "ERROR: imageDir must be a directory"
    exit
fi

newdir="$images/resize_${width}_${height}"
#echo $newdir

if [ ! -d "$newdir" ]; then
    mkdir $newdir
fi

mogrify -resize "${width}x${height}^" -gravity center -crop ${width}x${height}+0+0 +repage -path $newdir $images/*

echo "Done. Processed images are here: $newdir"
