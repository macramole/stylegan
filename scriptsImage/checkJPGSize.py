#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 02:06:59 2018

@author: macramole
"""
#%%

from PIL import Image
from glob import glob
import shutil
import os

#%%

imageDir = "/media/macramole/stuff/Data/open_images/Rose/"
targetWidth = 768
targetHeight = 768

files = glob(imageDir + "*.jpg")
badFiles = []

for f in files:
    im = Image.open(f)
    width, height = im.size
    
    if width < targetWidth or height < targetHeight:
        badFiles.append(f)
    
#%%
        
for f in badFiles:
    shutil.move(f, os.path.join(os.path.dirname(f), "badFiles", os.path.basename(f) ) )
    
#%%

for f in files:
    if f not in badFiles:
        shutil.move( f, os.path.join(os.path.dirname(f), "goodFiles", os.path.basename(f) ) )