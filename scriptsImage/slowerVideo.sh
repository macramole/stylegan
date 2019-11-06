ffmpeg -i out.mp4 -filter:v "setpts=2*PTS" result.mp4
