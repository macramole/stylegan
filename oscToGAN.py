#!/usr/bin/env python3

# -----------------------------------------------------------------------------
# Copyright (c) 2009-2016 Nicolas P. Rougier. All rights reserved.
# Distributed under the (new) BSD License.
# -----------------------------------------------------------------------------
from glumpy import app, gloo, gl
import numpy as np
import tensorflow as tf

import pickle

from pythonosc import dispatcher
from pythonosc import osc_server

PATH_MODEL = "/media/macramole/stuff/Data/sgan/bedrooms-256x256/karras2019stylegan-bedrooms-256x256.pkl"
defaultTruncation = 0.7

OUTPUT_RESOLUTION = None
SIZE_LATENT_SPACE = None

inputVector = None #esto va a  actualizarse por OSC
needToUpdateImage = True

OSC_IP = "127.0.0.1"
OSC_PORT = 4000

vertex = """
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
"""

fragment = """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
"""

def generateFromGAN(latents):
    # latents = np.random.RandomState(1000).randn(1000, *generator.input_shapes[0][1:]) # 1000 random latents
    # latents = latents[[477, 56, 83, 887, 583, 391, 86, 340, 341, 415]] # hand-picked top-10
    truncation = defaultTruncation
    
    # Generate dummy labels (not used by the official networks).
    labels = np.zeros([latents.shape[0]] + generator.input_shapes[1][1:])
    #fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    #images = generator.run(latents, labels, truncation_psi=0.7, randomize_noise=True ) #output_transform=fmt
    images = generator.run(latents, labels, truncation_psi=truncation, randomize_noise=False )

    # Convert images to PIL-compatible format.
    images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0, 255.0).astype(np.uint8) # [-1,1] => [0,255]
    images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

    return images

def initNeuralActivities():
    global generator, SIZE_LATENT_SPACE, OUTPUT_RESOLUTION, arrInterpolation, inputVector
    
    tf.InteractiveSession()
    with open( PATH_MODEL, 'rb' ) as file:
        _, _, generator = pickle.load(file)
        SIZE_LATENT_SPACE = int( generator.list_layers()[0][1].shape[1] )
        OUTPUT_RESOLUTION = int( generator.list_layers()[-1][1].shape[2] )
        
        print(f"Size latent space: {SIZE_LATENT_SPACE}") 
        print(f"Output resolution: {OUTPUT_RESOLUTION}")
    
    inputVector = np.random.uniform(-3, 3, (1, SIZE_LATENT_SPACE))

def initOSCServer():
    mdispatcher = dispatcher.Dispatcher()
    mdispatcher.map("/input", onOSCInputVector)
    
    server = osc_server.ThreadingOSCUDPServer( (OSC_IP, OSC_PORT), mdispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()

def onOSCInputVector( data ):
    global inputVector
    
    print(data)
        
def updateImage():
    global quad, currentImage, needToUpdateImage
    
    if needToUpdateImage:
        arrImage = generateFromGAN( inputVector )
        arrImage = np.ascontiguousarray(arrImage[0])    
        quad['texture'] = np.ascontiguousarray(arrImage)
        needToUpdateImage = False

initNeuralActivities()
initOSCServer()

window = app.Window(width=OUTPUT_RESOLUTION, height=OUTPUT_RESOLUTION, aspect=1)

@window.event
def on_draw(dt):
    window.clear()
    quad.draw(gl.GL_TRIANGLE_STRIP)
    updateImage()
    
quad = gloo.Program(vertex, fragment, count=4)
quad['position'] = [(-1,-1), (-1,+1), (+1,-1), (+1,+1)]
quad['texcoord'] = [( 0, 1), ( 0, 0), ( 1, 1), ( 1, 0)]

needToUpdateImage = True
updateImage()

app.run()
