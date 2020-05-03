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
from tkinter import simpledialog
from tkinter import messagebox
import numpy as np

import pickle
import tensorflow as tf
from PIL import Image, ImageTk
import subprocess

from scipy.interpolate import interp1d

generator = None
SIZE_LATENT_SPACE = None # filled on load
OUTPUT_RESOLUTION = None # filled on load

pointsSaved = []
inputVector = []

PATH_LOAD_FILE = "/media/macramole/stuff/Data/sgan/"
PATH_RESULT = "./generateResults/"
PATH_IMAGES_TO_VIDEO = "scriptsImage/imagesToVideo.sh"
PATH_IMAGES_TO_VIDEO_WITH_LOOP = "scriptsImage/imagesToVideoWithLoop.sh"

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
    # global generator
    tf.InteractiveSession()
    onLoadFile()

def generateFromGAN(latents):

    # latents = np.random.RandomState(1000).randn(1000, *generator.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
    truncation = defaultTruncation
    if sliderTruncation is not None:
        truncation = sliderTruncation.get()/10

    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + generator.input_shapes[1][1:])
    #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    #images = generator.run(latents, labels, truncation_psi=0.7, randomize_noise=True ) #output_transform=fmt
    images = generator.run(latents, labels, truncation_psi=truncation, randomize_noise=False )

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    return images

def generateImage():
    global inputVector

    if len(inputVector) == 0 :
        inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))

    img = generateFromGAN( inputVector )#noise)
    img = img[0]

    # photo = ImageTk.PhotoImage(image=Image.fromarray(img, 'RGB'))
    photo = Image.fromarray(img, 'RGB')
    return photo

def updateImage(newInputVector = None):
    global inputVector, recordingCurrentFrame, arrayImage

    if not type(newInputVector) is np.ndarray:
        # inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
        inputVector = np.random.uniform(-3, 3, (1, SIZE_LATENT_SPACE))
        drawAllPointsPair()
    else:
        inputVector = newInputVector

    arrayImage = generateImage()
    photo = ImageTk.PhotoImage(image=arrayImage)
    mainImage.configure(image=photo)
    mainImage.image = photo

    if recording:
        onAddPoint()

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

        inputVector[0][i] = mapValue(coords[0] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -3, 3)
        inputVector[0][i+1] = mapValue(coords[1] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -3, 3)

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
        x = mapValue(inputVector[0][p], -3,3, 0, OUTPUT_RESOLUTION)
        y = mapValue(inputVector[0][p+1], -3,3, 0, OUTPUT_RESOLUTION)

        if needToCreate:
            canvas.create_oval(x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS, fill=COLOR_POINT)
        else:
            canvas.coords( int(p/2) + 1, x-POINT_RADIUS,y-POINT_RADIUS,x+POINT_RADIUS,y+POINT_RADIUS )

def showFrame(generatedPhotos, index):
    photo = generatedPhotos[index]
    mainImage.configure(image=photo)
    mainImage.image = photo

    if index < 10:
        root.after(800, showFrame, generatedPhotos, index+1)

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


def onSaveVideo():
    global lastVideoFilename

    initialFilenameValue = modelPath.split("/")[-2] + "-00"
    if lastVideoFilename != None:
        if lastVideoFilename[-2].isdigit():
            lastNumber = int(lastVideoFilename[-2])
            lastNumber += 1
            initialFilenameValue = "%s-%d" % (lastVideoFilename[-2], lastNumber)
        else:
            initialFilenameValue = "%s-%d" % (lastVideoFilename, 0)

    videoFilename = simpledialog.askstring("Input", "Video filename:",
                                parent=root, initialvalue=initialFilenameValue)

    if videoFilename is None:
        return

    lastVideoFilename = videoFilename

    currentPathResult = os.path.join(PATH_RESULT, videoFilename)
    videoFilename += ".mp4"
    videoFilename = os.path.join(PATH_RESULT, videoFilename)

    try:
        os.mkdir(currentPathResult)
    except:
        messagebox.showerror("Error", "Path already exist")
        return

    btnSaveVideo.grid_remove()
    progressBar['value'] = 0
    progressBar.grid()

    root.update_idletasks()

    maxInterpolation = sliderTransition.get()
    cantInterpolations = calculateInterpolationPerPoint(maxInterpolation)
    totalFrames = np.sum(cantInterpolations)
    arrInterpolations = []

    for pointFrom in range(0, pointList.size()-1):
        pointTo = pointFrom + 1

        #loop
        if pointFrom == pointList.size()-1:
            pointTo = 0

        cantInterpolation = cantInterpolations[pointFrom]

        x = np.array([0,1])
        y = np.vstack((pointsSaved[pointFrom],pointsSaved[pointTo]))
        f = interp1d( x , y, axis = 0  )

        arrInterpolation = f( np.linspace(0,1,cantInterpolation+1, endpoint = True) )
        arrInterpolations += list(arrInterpolation)

    #batch_size = 20 512*512
    #batch_size = 10
    batch_size = int(sliderBatchSize.get())

    for i in range(0,len(arrInterpolations), batch_size):
        end = i + batch_size
        if end > len(arrInterpolations):
            end = len(arrInterpolations)

        latentSamples = np.array(arrInterpolations[i:end])
        generatedImages = generateFromGAN( latentSamples )

        for j, generated in enumerate(generatedImages):
            generatedImage = Image.fromarray(generated, 'RGB')
            currentFrame = i+j
            generatedImage.save( "%s/%05d.png" % (currentPathResult,currentFrame) )

            progressBar['value'] = (currentFrame/len(arrInterpolations))*100
            root.update_idletasks()

        # for i in range(0,len(arrInterpolation)):
            # if pointTo == 0 and i == len(arrInterpolation) - 1 :
            #     break

            # latentSample = np.array([arrInterpolation[i]])
            # generated = generateFromGAN( latentSample )
            # generatedImage = Image.fromarray(generated[0], 'RGB')
            # currentFrame = np.sum( cantInterpolations[0:pointFrom] )+i+1
            # generatedImage.save( "%s/%05d.png" % (currentPathResult,currentFrame) )


    script = PATH_IMAGES_TO_VIDEO if doLoop.get() == 0 else PATH_IMAGES_TO_VIDEO_WITH_LOOP
    subprocess.call([script, currentPathResult, videoFilename])

    progressBar.grid_remove()
    root.update_idletasks()
    btnSaveVideo.grid()
    root.update_idletasks()

    subprocess.call(["vlc", videoFilename])
        # generatedImages = generateFromGAN( arrInterpolation )
        # generatedPhotos = []
        #
        # for g in generatedImages:
        #     photo = ImageTk.PhotoImage(image=Image.fromarray(g, 'RGB'))
        #     generatedPhotos.append(photo)
        #
        # showFrame(generatedPhotos, 0)
def onSaveStill():
    global recordingCurrentFrame, stillFilename

    if stillFilename == None:
        stillFilename = simpledialog.askstring("Input", "Still directory name:",
                                parent=root, initialvalue="out")
    if stillFilename is None:
        return

    currentPathResult = os.path.join(PATH_RESULT, stillFilename)

    try:
        os.mkdir(currentPathResult)
    except:
        pass


    arrayImage.save( "%s/%05d.png" % (currentPathResult,recordingCurrentFrame) )
    recordingCurrentFrame += 1

def onRecord():
    global recording
    if not recording:
        recording = True
        btnRecord.config(text="Recording...")
    else :
        recording = False
        btnRecord.config(text="Record point dragging")

def onSliderTruncationChange(v):
    updateImage(inputVector)

def onRandomClick():
    global pointsMoveOriginalCoords
    pointsMoveOriginalCoords = None
    updateImage()

def onLoadFile():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, pointsSaved, modelPath

    modelPath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
    with open( modelPath, 'rb' ) as file:
        _, _, generator = pickle.load(file)
        SIZE_LATENT_SPACE = int( generator.list_layers()[0][1].shape[1] )
        OUTPUT_RESOLUTION = int( generator.list_layers()[-1][1].shape[2] )

        root.title('PGAN Generator - %s' % modelPath )

        # if canvas:
        #     canvas.delete("all")
        if pointList:
            pointList.delete(0,tk.END)
        pointsSaved = []
def onLoadFileMenu():
    onLoadFile()
    onRandomClick()



root = tk.Tk()
init()

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load file", command=onLoadFileMenu)
menubar.add_cascade(label="File", menu=filemenu)

root.config(menu=menubar)

canvas = tk.Canvas(root,width=OUTPUT_RESOLUTION, height=OUTPUT_RESOLUTION, bg="#000000", cursor="cross")
canvas.grid(row=0,column=1)
canvas.bind("<B1-Motion>", xyMoved)
canvas.bind("<ButtonRelease-1>", xyMovedFinished)
canvas.bind("<Button 3>", xyClicked)

arrayImage = generateImage()
photo = ImageTk.PhotoImage(image=arrayImage)
drawAllPointsPair()
mainImage = tk.Label(root, image=photo)
mainImage.image = photo #esto es necesario por el garbage
# mainImage.pack( padx = SIZE_LATENT_SPACE)
mainImage.grid(row=0,column=2)

btnSaveStill = tk.Button(root, text="Save still", command=onSaveStill)
btnSaveStill.grid(row=1,column=2)


optionsFrame = tk.Frame(root)
optionsFrame.grid(row=1, column=1)
btnRandom = tk.Button(optionsFrame, text="Random", command=onRandomClick)
btnRandom.pack()
lblTruncation = tk.Label(optionsFrame, text='Truncation:')
lblTruncation.pack(side = tk.LEFT)
sliderTruncation = tk.Scale(optionsFrame, from_=-30, to=30, resolution=1, orient=tk.HORIZONTAL, command=onSliderTruncationChange)
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
