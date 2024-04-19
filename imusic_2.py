import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pandas as pd
import IPython.display as ipd
import librosa
from midiutil import MIDIFile
import random
from pedalboard import Pedalboard, Chorus, Reverb
from pedalboard.io import AudioFile
from scipy.io import wavfile

#Get height and width of image
# height, width, _ = img.shape
#Convert a hue value to wavelength via interpolation
#Assume that visible spectrum is contained between 400-650nm
def hue2wl(h, wlMax = 650, wlMin = 400, hMax = 270, hMin = 0):
    #h *= 2
    hMax /= 2
    hMin /= 2
    wlRange = wlMax - wlMin
    hRange = hMax - hMin
    wl =  wlMax - ((h* (wlRange))/(hRange))
    return wl

def wl2freq(wl):
    wavelength = wl
    sol = 299792458.00 #this is the speed of light in m/s
    sol *= 1e9 #Convert speed of light to nm/s
    freq = (sol / wavelength) * (1e-12)
    return freq

def img2music2(img, height,width):



    #Convet from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #Populate hues array with H channel for each pixel
    i=0 ; j=0
    hues = []
    for i in range(height):
        for j in range(width):
            hue = hsv[i][j][0] #This is the hue value at pixel coordinate (i,j)
            hues.append(hue)

    #Make pandas dataframe
    hues_df = pd.DataFrame(hues, columns=['hues'])
    hues_df['nm'] = hues_df.apply(lambda row : hue2wl(row['hues']), axis = 1)
    hues_df['freq'] = hues_df.apply(lambda row : wl2freq(row['nm']), axis = 1)
    hues_df['notes'] = hues_df.apply(lambda row : librosa.hz_to_note(row['freq']), axis = 1)
    hues_df['midi_number'] = hues_df.apply(lambda row : librosa.note_to_midi(row['notes']), axis = 1)

    print("Done making song from image!")

    return hues_df


def make_music(name, img, height, width):
    huesdf = img2music2(img,height, width)

    song = huesdf['freq'].to_numpy()



    frequencies = huesdf['freq'].to_numpy()
    song = np.array([])
    harmony = np.array([])
    octaves = np.array([1/4,1,2,1,2])
    sr = 22050 # sample rate
    T = 0.5   # 0.1 second duration
    t = np.linspace(0, T, int(T*sr), endpoint=False) # time variable
    #Make a song with numpy array :]
    nPixels = int(len(frequencies)/height)
    # nPixels = 1000
    #for j in tqdm(range(nPixels), desc="Processing Frame"):#Add progress bar for frames processed
    for i in range(nPixels):
        octave = random.choice(octaves)
        val =  octave * frequencies[i]
        note  = 0.5*np.sin(2*np.pi*val*t)
        song  = np.concatenate([song, note])

    wavfile.write(make_music.__name__ + "_" + name  + '.wav'    , rate = 22050, data = song.astype(np.float32))



