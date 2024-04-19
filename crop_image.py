import numpy as np
import cv2
import matplotlib.pyplot as plt
from imusic_2 import make_music
from imusic import make_tunes
import os

#Options for creating a harmony
harmony = {'U0',  'ST','M2','m3','M3','P4','DT','P5','m6', 'M6',
                      'm7', 'M7', 'O8'  }
Keys = ['A', 'a', 'B', 'C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g']
Scales = ['AEOLIAN','BLUES', 'LYDIAN', 'CHROMATIC', 'HARMONIC_MINOR','DIATONIC_MINOR', 'PHYRIGIAN','MAJOR', 'DORIA'
          'HARMONIC_MINOR','MINOR', 'MELODIC_MINOR', 'MIXOLYDIAN']

octave = np.array([1,2,4,6,7,5, 1, 2])
#Options for creating a harmony
def downloadImage(URL):
    """Downloads the image on the URL, and convers to cv2 BGR format"""
    from io import BytesIO
    from PIL import Image as PIL_Image
    import requests

    response = requests.get(URL)
    image = PIL_Image.open(BytesIO(response.content))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)


fig, ax = plt.subplots()

image_name = 'Noah.jpg'
img =  cv2.imread(image_name)
name = image_name.split(".")[-1]
plt.imshow(img)
selectedRectangle = None
# Capture mouse-drawing-rectangle event
def line_select_callback(eclick, erelease):
    x1, y1 = eclick.xdata, eclick.ydata
    x2, y2 = erelease.xdata, erelease.ydata

    selectedRectangle = plt.Rectangle(
        (min(x1, x2), min(y1, y2)),
        np.abs(x1 - x2),
        np.abs(y1 - y2),
        color="r",
        fill=False,
    )
    ax.add_patch(selectedRectangle)

    imgOnRectangle = img[
        int(min(y1, y2)) : int(max(y1, y2)), int(min(x1, x2)) : int(max(x1, x2))
    ]
    height, width, _ = imgOnRectangle.shape
    make_tunes(name,imgOnRectangle, oct = None, key = None, sle = None, har = None )
    make_music(name, imgOnRectangle, height,width)



    plt.figure()
    plt.imshow(imgOnRectangle)
    plt.show()


from matplotlib.widgets import RectangleSelector

rs = RectangleSelector(
    ax,
    line_select_callback,

    useblit=False,
    button=[1],
    minspanx=5,
    minspany=5,
    spancoords="pixels",
    interactive=True,
)

plt.show()
