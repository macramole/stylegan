#!/usr/bin/env python
import argparse
import os
from glob import glob
import subprocess

FILE_TYPE = "png"
OUT_DIR = "resultCollage"

parser = argparse.ArgumentParser()
parser.add_argument("dir1", help="Directory with frames")
parser.add_argument("dir2", help="Directory with frames")
parser.add_argument("dir3", help="Directory with frames")
args = parser.parse_args()

try:
    os.mkdir(OUT_DIR, False)
except:
    pass

for file in glob( os.path.join(args.dir1, "*.%s" % FILE_TYPE) ):
    fileName = file[ file.rfind("/")+1: ]

    file1 = os.path.join(args.dir1, fileName)
    file2 = os.path.join(args.dir2, fileName)
    file3 = os.path.join(args.dir3, fileName)
    out = os.path.join(OUT_DIR, fileName)

    subprocess.call(["montage", file1, file2, file3, "-tile", "1x", "-gravity", "North", "-geometry", "+0+4", "-background", "black", out])
