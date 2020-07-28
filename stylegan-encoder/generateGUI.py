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
from functools import partial

import numpy as np

import pickle
import tensorflow as tf
from PIL import Image, ImageTk
import subprocess

from encoder.generator_model import Generator

from scipy.interpolate import interp1d

encoderGenerator = None # new model capable of loading mappings (1, 18, 512)
generator = None
SIZE_LATENT_SPACE = None # filled on load
OUTPUT_RESOLUTION = None # filled on load

currentShowingDim = 0

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
	print(latents.shape)
	if latents.shape[0] != 18:
		truncation = defaultTruncation
		if sliderTruncation is not None:
			truncation = sliderTruncation.get()/10

		# Generate dummy labels (not used by the official networks).
		labels = np.zeros([latents.shape[0]] + generator.input_shapes[1][1:])
		images = generator.run(latents, labels, truncation_psi=truncation, randomize_noise=False )

		# Convert images to PIL-compatible format.
		images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
		images = images.transpose(0, 2, 3, 1) # NCHW => NHWC
	else:
		print("using encoder generator")
		latent_vector = latents.reshape((1, 18, 512))
		encoderGenerator.set_dlatents(latent_vector)
		images = encoderGenerator.generate_images()

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
		inputVector = np.random.normal(0, 1, (1, SIZE_LATENT_SPACE))
		# inputVector = np.random.uniform(-3, 3, (1, SIZE_LATENT_SPACE))
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

		inputVector[currentShowingDim][i] = mapValue(coords[0] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -0.5, 0.5)
		inputVector[currentShowingDim][i+1] = mapValue(coords[1] + POINT_RADIUS, 0, OUTPUT_RESOLUTION, -0.5, 0.5)

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

	for p in range(0, inputVector[currentShowingDim].shape[0] - 1, 2):
		x = mapValue(inputVector[currentShowingDim][p], -0.5,0.5, 0, OUTPUT_RESOLUTION)
		y = mapValue(inputVector[currentShowingDim][p+1], -0.5,0.5, 0, OUTPUT_RESOLUTION)

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

	initialFilenameNumber = 0
	modelName = modelPath.split("/")[-2] if lastVideoFilename == None else lastVideoFilename[0:-3]
	initialFilenameValue = modelName + "-{:0>2d}".format(initialFilenameNumber)
	while ( os.path.exists( os.path.join(PATH_RESULT, initialFilenameValue) ) ):
		initialFilenameNumber += 1
		initialFilenameValue = modelName + "-{:0>2d}".format(initialFilenameNumber)

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

	script = PATH_IMAGES_TO_VIDEO if doLoop.get() == 0 else PATH_IMAGES_TO_VIDEO_WITH_LOOP
	subprocess.call([script, currentPathResult, videoFilename])

	progressBar.grid_remove()
	root.update_idletasks()
	btnSaveVideo.grid()
	root.update_idletasks()

	subprocess.call(["vlc", videoFilename])

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
	global generator, encoderGenerator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, pointsSaved, modelPath

	# modelPath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
	modelPath = "/media/leandro/stuff/Data/ahegao/network-snapshot-011225.pkl"
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

		encoderGenerator = Generator(generator, 1)

def onLoadLatentVectorFromFile():
	latentPath = filedialog.askopenfilename(initialdir = PATH_LOAD_FILE, title = "Select file")
	latent_vector = np.load(latentPath)
	pointList.insert(tk.END, "%d" % ( pointList.size()+1 ) )
	pointsSaved.append( np.copy( latent_vector ) )

	updateImage( np.copy(pointsSaved[pointList.size()-1]) )
	drawAllPointsPair()

def onLoadFileMenu():
	onLoadFile()
	onRandomClick()

def onLatentDimClick(*args):
	global currentShowingDim
	currentShowingDim = int(tkVarDimension.get())
	drawAllPointsPair()

def onRandomDimensionClick():
	global inputVector
	print(sliderRandomDimension.get())
	inputVector[currentShowingDim] = np.random.normal(0, sliderRandomDimension.get(), (1, SIZE_LATENT_SPACE))
	updateImage(inputVector)
	drawAllPointsPair()

root = tk.Tk()
init()

menubar = tk.Menu(root)
filemenu = tk.Menu(menubar, tearoff=0)
filemenu.add_command(label="Load model", command=onLoadFileMenu)
filemenu.add_command(label="Load latent vector", command=onLoadLatentVectorFromFile)

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

tkVarDimension = tk.StringVar(root)
dimensionChoices = {}
for i in range(18):
	dimensionChoices[str(i)] = i
tkVarDimension.set(0)
dimensionOptionMenu = tk.OptionMenu(optionsFrame, tkVarDimension, *dimensionChoices)
# Label(mainframe, text="Choose a dish").grid(row = 1, column = 1)
# popupMenu.grid(row = 2, column =1)
tkVarDimension.trace('w', onLatentDimClick)
dimensionOptionMenu.pack()

btnRandomDimension = tk.Button(optionsFrame, text="Random dimension", command=onRandomDimensionClick)
btnRandomDimension.pack(side = tk.LEFT)
sliderRandomDimension =  tk.Scale(optionsFrame, from_=0.01, to=0.3, resolution=0.01, orient=tk.HORIZONTAL)
sliderRandomDimension.pack(side = tk.LEFT, fill=tk.X)

btnRandom = tk.Button(optionsFrame, text="Random", command=onRandomClick)
btnRandom.pack(fill=tk.Y)
# lblTruncation = tk.Label(optionsFrame, text='Truncation:')
# lblTruncation.pack(side = tk.LEFT)
# sliderTruncation = tk.Scale(optionsFrame, from_=-30, to=30, resolution=1, orient=tk.HORIZONTAL, command=onSliderTruncationChange)
# sliderTruncation.set( int(defaultTruncation * 10) )
# sliderTruncation.pack(side = tk.LEFT, fill=tk.X)

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
btnSaveVideo = tk.Button(root, text="Save video", command=onSaveVideo)
btnSaveVideo.grid(row=1, column=3)


root.mainloop()
