#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  7 14:35:21 2018

@author: leandro
"""

#%%

import os
import tkinter as tk
import tkinter.ttk as ttk
from tkinter import filedialog
import numpy as np
from pythonosc import udp_client

generator = None
SIZE_LATENT_SPACE = 512 # esto quizas deberia queriar de oscToGAN ?
OUTPUT_RESOLUTION = 256 # esto quizas deberia queriar de oscToGAN ?
CANVAS_SIZE = 512

OSC_IP = "127.0.0.1"
OSC_PORT = 4000
oscClient = None

pointsSaved = []
inputVector = []

PATH_LOAD_FILE = "/media/macramole/stuff/Data/sgan/"

lastX = 0
lastY = 0

selectionRectangle = None
selectionRectangleOriginalCoords = None
pointsMoveOriginalCoords = None
COLOR_POINT = "green"
COLOR_SELECTED = "red"
POINT_RADIUS = 2

canvas = None
pointList = None

recording = False
recordingCurrentFrame = 0

stillFilename = None
lastVideoFilename = None
modelPath = None

arrayImage = None

defaultTruncation = 0.7
sliderTruncation = None #defined later
#%%

def init():
    global oscClient
    
    oscClient = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)

def xyClicked(e):
    global pointsMoveOriginalCoords

    pointsMoveOriginalCoords = None
    canvas.itemconfig( "selected", fill=COLOR_POINT )
    canvas.dtag("selected")
def xyMoved(e):
    global selectionRectangle, selectionRectangleOriginalCoords, pointsMoveOriginalCoords

    if selectionRectangle is None:

        if len( canvas.find_withtag("selected") ) == 0:
            selectionRectangle = canvas.create_rectangle(e.x, e.y, e.x + 5, e.y + 5, outline = "white")
            selectionRectangleOriginalCoords = (e.x, e.y)
        else:
            if pointsMoveOriginalCoords is None:
                pointsMoveOriginalCoords = (e.x, e.y)
            else:
                canvas.move("selected", e.x - pointsMoveOriginalCoords[0], e.y - pointsMoveOriginalCoords[1])
                pointsMoveOriginalCoords = (e.x, e.y)
                pointsMoved()

    else:
        x0 = selectionRectangleOriginalCoords[0]
        y0 = selectionRectangleOriginalCoords[1]
        x1 = e.x
        y1 = e.y
        canvas.coords(selectionRectangle, x0, y0, x1, y1)

        canvas.itemconfig( "selected", fill=COLOR_POINT )
        canvas.dtag("selected")
        canvas.addtag_overlapping("selected", x0, y0, x1, y1 )
        canvas.dtag(selectionRectangle, "selected")
        canvas.itemconfig( "selected", fill=COLOR_SELECTED )
def xyMovedFinished(e):
    global selectionRectangle, pointsMoveOriginalCoords

    canvas.delete(selectionRectangle)
    selectionRectangle = None
    pointsMoveOriginalCoords = None

def pointsMoved():
    global inputVector

    for p in canvas.find_withtag("selected"):
        i = (p - 1) * 2
        coords = canvas.coords(p)

        inputVector[0][i] = mapValue(coords[0] + POINT_RADIUS, 0, CANVAS_SIZE, -3, 3)
        inputVector[0][i+1] = mapValue(coords[1] + POINT_RADIUS, 0, CANVAS_SIZE, -3, 3)

    updateImage(inputVector)

def mapValue(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

def onAddPoint():
    # pointList.insert(END, "%f,%f" % ( inputVector[0][0], inputVector[0][1] ) )
    pointList.insert(tk.END, "%d" % ( pointList.size()+1 ) )
    pointsSaved.append( np.copy( inputVector ) )

def onRemovePoints():
    global pointsSaved

    pointList.delete(0,tk.END)
    pointsSaved = []

def onRemovePoint():
    global pointsSaved

    pointList.delete(tk.END)
    pointsSaved.pop()

def onListClicked(e):
    if recording:
        onRecord()
    updateImage( np.copy(pointsSaved[pointList.curselection()[0]]) )
    drawAllPointsPair()

def drawAllPointsPair():
    needToCreate = True
    if len(canvas.find_all()) > 0:
        needToCreate = False
        # canvas.itemconfig( "selected", fill = COLOR_POINT )
        # canvas.dtag( "selected" )

    for p in range(0, inputVector[0].shape[0] - 1, 2):
        x = mapValue(inputVector[0][p], -3,3, 0, CANVAS_SIZE)
        y = mapValue(inputVector[0][p+1], -3,3, 0, CANVAS_SIZE)

        if needToCreate:
            canvas.create_oval(x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS, fill=COLOR_POINT)
        else:
            canvas.coords( int(p/2) + 1, x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS )

def calculateDistances():
    distances = []
    for pointFrom in range(0, pointList.size()):
        pointTo = pointFrom + 1

        #loop
        if pointFrom == pointList.size()-1:
            pointTo = 0

        dist = np.linalg.norm(pointsSaved[pointTo]-pointsSaved[pointFrom])
        distances.append(dist)

    return distances

def calculateInterpolationPerPoint(maxInterpolation):
    distances = calculateDistances()
    maxDistance = np.max(distances)
    cantInterpolations = []

    for pointFrom in range(0, pointList.size()):
        cantInterpolation = mapValue( distances[pointFrom], 0, maxDistance, 1, maxInterpolation )
        cantInterpolation = int(np.floor(cantInterpolation))
        cantInterpolations.append(cantInterpolation)

    return cantInterpolations

def onRecord():
    global recording
    if not recording:
        recording = True
        btnRecord.config(text="Recording...")
    else :
        recording = False
        btnRecord.config(text="Record point dragging")

def updateImage(newInputVector = None):
    global inputVector, recordingCurrentFrame, arrayImage

    if not type(newInputVector) is np.ndarray:
        # inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
        inputVector = np.random.uniform(-3, 3, (1, SIZE_LATENT_SPACE))
        drawAllPointsPair()
    else:
        inputVector = newInputVector

    if recording:
        onAddPoint()
        
    # print( "update" )
    # print( list(inputVector) )
    # print( list(inputVector[0]) )
    
    oscClient.send_message("/input", list(inputVector[0]))
    # oscClient.send_message("/input", 1)

def onSliderTruncationChange(v):
    updateImage(inputVector)

def onRandomClick():
    global pointsMoveOriginalCoords
    pointsMoveOriginalCoords = None
    updateImage()

def onLoadFile():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, pointsSaved, modelPath

    modelPath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
    
    if pointList:
        pointList.delete(0,tk.END)
    pointsSaved = []
    
def onLoadFileMenu():
    onLoadFile()
    onRandomClick()

def onSaveVideo():
    pass

def onSaveStill():
    pass

root = tk.Tk()
init()

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load file", command=onLoadFileMenu)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)

canvas = tk.Canvas(root,width=CANVAS_SIZE, height=CANVAS_SIZE, bg="#000000", cursor="cross")
canvas.grid(row=0,column=1)
canvas.bind("<B1-Motion>", xyMoved)
canvas.bind("<ButtonRelease-1>", xyMovedFinished)
canvas.bind("<Button 3>", xyClicked)

updateImage()
drawAllPointsPair()

btnSaveStill = tk.Button(root, text="Save still", command=onSaveStill)
btnSaveStill.grid(row=1,column=2)

optionsFrame = tk.Frame(root)
optionsFrame.grid(row=1, column=1)
btnRandom = tk.Button(optionsFrame, text="Random", command=onRandomClick)
btnRandom.pack()
lblTruncation = tk.Label(optionsFrame, text='Truncation:')
lblTruncation.pack(side = tk.LEFT)
sliderTruncation = tk.Scale(optionsFrame, to=10, resolution=1, orient=tk.HORIZONTAL, command=onSliderTruncationChange)
sliderTruncation.set( int(defaultTruncation * 10) )
sliderTruncation.pack(side = tk.LEFT, fill=tk.X)

pointsFrame = tk.Frame(root)
pointsFrame.grid(row=0,column=3, padx = 5)

btnRecord = tk.Button(pointsFrame, text="Record point dragging", command=onRecord)
btnRecord.pack()
btnAddPoint = tk.Button(pointsFrame, text= "Add point", command=onAddPoint)
btnAddPoint.pack()
pointList = tk.Listbox(pointsFrame, height = 20, justify=tk.CENTER)
pointList.pack()
pointList.bind("<Double-Button-1>", onListClicked)
btnRmPoint = tk.Button(pointsFrame, text= "Remove last point", command=onRemovePoint)
btnRmPoint.pack()
btnRmPoints = tk.Button(pointsFrame, text= "Remove all points", command=onRemovePoints)
btnRmPoints.pack()
tk.Label(pointsFrame, text='').pack() #spacer
lblTransition = tk.Label(pointsFrame, text='Max transition length:')
lblTransition.pack()
sliderTransition = tk.Scale(pointsFrame, from_=5, to=1000, resolution=5, orient=tk.HORIZONTAL)
sliderTransition.set(25)
sliderTransition.pack(fill=tk.X)
sliderBatchSize = tk.Scale(pointsFrame, from_=1, to=100, resolution=1, orient=tk.HORIZONTAL)
lblBatchSize = tk.Label(pointsFrame, text='Batch size:')
lblBatchSize.pack()
sliderBatchSize.set(5)
sliderBatchSize.pack(fill=tk.X)

doLoop = tk.IntVar()
chkLoop = tk.Checkbutton(pointsFrame, text="Loop video", variable=doLoop)
chkLoop.pack()

progressBar = ttk.Progressbar(root,orient=tk.HORIZONTAL,length=100,mode='determinate')
# progressBar.pack()
progressBar.grid(row=1, column=3)
progressBar.grid_remove()
btnSaveVideo = tk.Button(root, text= "Save video", command=onSaveVideo)
btnSaveVideo.grid(row=1, column=3)

root.mainloop()
