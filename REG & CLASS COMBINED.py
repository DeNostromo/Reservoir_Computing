# %% REG & CLASS COMBINED 

os.chdir(r"C:\Users\Sixten\OneDrive - Danmarks Tekniske Universitet\13. semester (sidste)\Speciale\Scripts")

import os
import sys
import importlib
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import random as rnd
import time
from sklearn.model_selection import train_test_split
import pesq
import scipy.io
import random
from scipy.signal import resample
from sklearn.metrics import accuracy_score

from scipy.signal import stft, istft

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.utils 
import keras
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import Ridge
from reservoirpy.nodes import Input

import functions
importlib.reload(functions)
import Prepare_data
import ESN
import evaluate

import librosa

importlib.reload(ESN)
importlib.reload(evaluate)
importlib.reload(Prepare_data)


 # %%

# set main path 
path = r"D:\Speciale\data\\"

# sampling freq 
fs = 16000

# Type of noise 
noise_type = "white_noise"

# Model type
model_type = "CNN"

snr_list = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
snr_list = [10]

#snr_list = [-10]

# Noise factor
noise_factor = 0.05

#Segment length in samples  
#segment_length = 1024*32 # 1024 er 64 ms 2 SEK  
segment_length = 1024*32 # 1024 er 64 ms 2 SEK  

print("segment length: ", segment_length/fs, "seconds")


# %% PREPARE DATA

for i in range(len(snr_list)):

    snr = snr_list[i]
    
    Prepare_data.make_data(path, noise_factor, segment_length, noise_type, snr)



# %% HRIR

snr = 10 
# Convolve data 

# Load the .mat file
data = scipy.io.loadmat(r"C:\Users\Sixten\Documents\Speciale\hrir_final.mat")

hrir_l_data = data["hrir_l"]
hrir_r_data = data["hrir_r"]
#hrir_l_data = hrir_l_data[:][0][:]


# Azimuth vector
azimuth = np.concatenate([
    np.array([-80, -65, -55]),  # Specific values
    np.arange(-45, 46, 5),      # Range from -45 to 45 with step 5
    np.array([55, 65, 80])      # Specific values
])
print(azimuth)

# Elevation vector 
elevation = -45 + 5.625 * np.arange(0, 50)  # 0:49 in MATLAB is 0 to 49 (exclusive of 50)
print(elevation)

# Extraft HRIR
def extract_hrir(hrir_l, hrir_r):
    # -80 deg
    hrir_l_m80 = hrir_l[0,8,:]
    hrir_r_m80 = hrir_r[0,8,:]
    hrir_l_m80 = librosa.resample(hrir_l_m80, orig_sr=44100, target_sr=16000)
    hrir_r_m80 = librosa.resample(hrir_r_m80, orig_sr=44100, target_sr=16000)
    hrir_m80 = np.vstack((hrir_l_m80, hrir_r_m80))
    # -65 deg
    hrir_l_m65 = hrir_l[1,8,:]
    hrir_r_m65 = hrir_r[1,8,:]
    hrir_l_m65 = librosa.resample(hrir_l_m65, orig_sr=44100, target_sr=16000)
    hrir_r_m65 = librosa.resample(hrir_r_m65, orig_sr=44100, target_sr=16000)
    hrir_m65 = np.vstack((hrir_l_m65, hrir_r_m65))
    # -55 deg
    hrir_l_m55 = hrir_l[2,8,:]
    hrir_r_m55 = hrir_r[2,8,:]
    hrir_l_m55 = librosa.resample(hrir_l_m55, orig_sr=44100, target_sr=16000)
    hrir_r_m55 = librosa.resample(hrir_r_m55, orig_sr=44100, target_sr=16000)
    hrir_m55 = np.vstack((hrir_l_m55, hrir_r_m55))
    # -45 deg
    hrir_l_m45 = hrir_l[3,8,:]
    hrir_r_m45 = hrir_r[3,8,:]
    hrir_l_m45 = librosa.resample(hrir_l_m45, orig_sr=44100, target_sr=16000)
    hrir_r_m45 = librosa.resample(hrir_r_m45, orig_sr=44100, target_sr=16000)
    hrir_m45 = np.vstack((hrir_l_m45, hrir_r_m45))
    # -40 deg
    hrir_l_m40 = hrir_l[4,8,:]
    hrir_r_m40 = hrir_r[4,8,:]
    hrir_l_m40 = librosa.resample(hrir_l_m40,orig_sr= 44100, target_sr=16000)
    hrir_r_m40 = librosa.resample(hrir_r_m40, orig_sr=44100, target_sr=16000)
    hrir_m40 = np.vstack((hrir_l_m40, hrir_r_m40))
    # -35 deg
    hrir_l_m35 = hrir_l[5,8,:]
    hrir_r_m35 = hrir_r[5,8,:]
    hrir_l_m35 = librosa.resample(hrir_l_m35, orig_sr=44100, target_sr=16000)
    hrir_r_m35 = librosa.resample(hrir_r_m35, orig_sr=44100, target_sr=16000)
    hrir_m35 = np.vstack((hrir_l_m35, hrir_r_m35))
    # -30 deg
    hrir_l_m30 = hrir_l[6,8,:]
    hrir_r_m30 = hrir_r[6,8,:]
    hrir_l_m30 = librosa.resample(hrir_l_m30, orig_sr=44100, target_sr=16000)
    hrir_r_m30 = librosa.resample(hrir_r_m30, orig_sr=44100, target_sr=16000)
    hrir_m30 = np.vstack((hrir_l_m30, hrir_r_m30))
    # -25 deg
    hrir_l_m25 = hrir_l[7,8,:]
    hrir_r_m25 = hrir_r[7,8,:]
    hrir_l_m25 = librosa.resample(hrir_l_m25, orig_sr=44100, target_sr=16000)
    hrir_r_m25 = librosa.resample(hrir_r_m25, orig_sr=44100, target_sr=16000)
    hrir_m25 = np.vstack((hrir_l_m25, hrir_r_m25))
    # -20 deg
    hrir_l_m20 = hrir_l[8,8,:]
    hrir_r_m20 = hrir_r[8,8,:]
    hrir_l_m20 = librosa.resample(hrir_l_m20, orig_sr=44100, target_sr=16000)
    hrir_r_m20 = librosa.resample(hrir_r_m20, orig_sr=44100, target_sr=16000)
    hrir_m20 = np.vstack((hrir_l_m20, hrir_r_m20))
    # -15 deg
    hrir_l_m15 = hrir_l[9,8,:]
    hrir_r_m15 = hrir_r[9,8,:]
    hrir_l_m15 = librosa.resample(hrir_l_m15, orig_sr=44100, target_sr=16000)
    hrir_r_m15 = librosa.resample(hrir_r_m15, orig_sr=44100, target_sr=16000)
    hrir_m15 = np.vstack((hrir_l_m15, hrir_r_m15))
    # -10 deg
    hrir_l_m10 = hrir_l[10,8,:]
    hrir_r_m10 = hrir_r[10,8,:]
    hrir_l_m10 = librosa.resample(hrir_l_m10, orig_sr=44100, target_sr=16000)
    hrir_r_m10 = librosa.resample(hrir_r_m10, orig_sr=44100, target_sr=16000)
    hrir_m10 = np.vstack((hrir_l_m10, hrir_r_m10))
    # -5 deg
    hrir_l_m5 = hrir_l[11,8,:]
    hrir_r_m5 = hrir_r[11,8,:]
    hrir_l_m5 = librosa.resample(hrir_l_m5, orig_sr=44100, target_sr=16000)
    hrir_r_m5 = librosa.resample(hrir_r_m5, orig_sr=44100, target_sr=16000)
    hrir_m5 = np.vstack((hrir_l_m5, hrir_r_m5))
    # 0 deg 
    hrir_l_0 = hrir_l[12,8,:]
    hrir_r_0 = hrir_r[12,8,:]
    hrir_l_0 = librosa.resample(hrir_l_0, orig_sr=44100, target_sr=16000)
    hrir_r_0 = librosa.resample(hrir_r_0, orig_sr=44100, target_sr=16000)
    hrir_0 = np.vstack((hrir_l_0, hrir_r_0))
    # 5 deg 
    hrir_l_5 = hrir_l[13,8,:]
    hrir_r_5 = hrir_r[13,8,:]
    hrir_l_5 = librosa.resample(hrir_l_5, orig_sr=44100, target_sr=16000)
    hrir_r_5 = librosa.resample(hrir_r_5, orig_sr=44100, target_sr=16000)
    hrir_5 = np.vstack((hrir_l_5, hrir_r_5))
    # 10 deg 
    hrir_l_10 = hrir_l[14,8,:]
    hrir_r_10 = hrir_r[14,8,:]
    hrir_l_10 = librosa.resample(hrir_l_10, orig_sr=44100, target_sr=16000)
    hrir_r_10 = librosa.resample(hrir_r_10, orig_sr=44100, target_sr=16000)
    hrir_10 = np.vstack((hrir_l_10, hrir_r_10))
    # 15 deg 
    hrir_l_15 = hrir_l[15,8,:]
    hrir_r_15 = hrir_r[15,8,:]
    hrir_l_15 = librosa.resample(hrir_l_15, orig_sr=44100, target_sr=16000)
    hrir_r_15 = librosa.resample(hrir_r_15, orig_sr=44100, target_sr=16000)
    hrir_15 = np.vstack((hrir_l_15, hrir_r_15))
    # 20 deg 
    hrir_l_20 = hrir_l[16,8,:]
    hrir_r_20 = hrir_r[16,8,:]
    hrir_l_20 = librosa.resample(hrir_l_20, orig_sr=44100, target_sr=16000)
    hrir_r_20 = librosa.resample(hrir_r_20, orig_sr=44100, target_sr=16000)
    hrir_20 = np.vstack((hrir_l_20, hrir_r_20))
    # 25 deg 
    hrir_l_25 = hrir_l[17,8,:]
    hrir_r_25 = hrir_r[17,8,:]
    hrir_l_25 = librosa.resample(hrir_l_25, orig_sr=44100, target_sr=16000)
    hrir_r_25 = librosa.resample(hrir_r_25, orig_sr=44100, target_sr=16000)
    hrir_25 = np.vstack((hrir_l_25, hrir_r_25))
    # 30 deg 
    hrir_l_30 = hrir_l[18,8,:]
    hrir_r_30 = hrir_r[18,8,:]
    hrir_l_30  = librosa.resample(hrir_l_30 , orig_sr=44100, target_sr=16000)
    hrir_r_30  = librosa.resample(hrir_r_30 , orig_sr=44100, target_sr=16000)
    hrir_30 = np.vstack((hrir_l_30, hrir_r_30))
    # 35 deg 
    hrir_l_35 = hrir_l[19,8,:]
    hrir_r_35 = hrir_r[19,8,:]
    hrir_l_35 = librosa.resample(hrir_l_35, orig_sr=44100, target_sr=16000)
    hrir_r_35 = librosa.resample(hrir_r_35, orig_sr=44100, target_sr=16000)
    hrir_35 = np.vstack((hrir_l_35, hrir_r_35))
    # 40 deg 
    hrir_l_40 = hrir_l[20,8,:]
    hrir_r_40 = hrir_r[20,8,:]
    hrir_l_40 = librosa.resample(hrir_l_40, orig_sr=44100, target_sr=16000)
    hrir_r_40 = librosa.resample(hrir_r_40, orig_sr=44100, target_sr=16000)
    hrir_40 = np.vstack((hrir_l_40, hrir_r_40))
    # 45 deg 
    hrir_l_45 = hrir_l[21,8,:]
    hrir_r_45 = hrir_r[21,8,:]
    hrir_l_45 = librosa.resample(hrir_l_45, orig_sr=44100, target_sr=16000)
    hrir_r_45 = librosa.resample(hrir_r_45, orig_sr=44100, target_sr=16000)
    hrir_45 = np.vstack((hrir_l_45, hrir_r_45))
    # 55 deg 
    hrir_l_55 = hrir_l[22,8,:]
    hrir_r_55 = hrir_r[22,8,:]
    hrir_l_55 = librosa.resample(hrir_l_55, orig_sr=44100, target_sr=16000)
    hrir_r_55 = librosa.resample(hrir_r_55, orig_sr=44100, target_sr=16000)
    hrir_55 = np.vstack((hrir_l_55, hrir_r_55))
    # 65 deg 
    hrir_l_65 = hrir_l[23,8,:]
    hrir_r_65 = hrir_r[23,8,:]
    hrir_l_65 = librosa.resample(hrir_l_65, orig_sr=44100, target_sr=16000)
    hrir_r_65 = librosa.resample(hrir_r_65, orig_sr=44100, target_sr=16000)
    hrir_65 = np.vstack((hrir_l_65, hrir_r_65))
    # 80 deg
    hrir_l_80 = hrir_l[24,8,:]
    hrir_r_80 = hrir_r[24,8,:]
    hrir_l_80 = librosa.resample(hrir_l_80, orig_sr=44100, target_sr=16000)
    hrir_r_80 = librosa.resample(hrir_r_80, orig_sr=44100, target_sr=16000)
    hrir_80 = np.vstack((hrir_l_80, hrir_r_80))

    # For back of sphere 

    # -80 deg
    hrir_l_m80 = hrir_l[0,40,:]
    hrir_r_m80 = hrir_r[0,40,:]
    hrir_l_m80 = librosa.resample(hrir_l_m80, orig_sr=44100, target_sr=16000)
    hrir_r_m80 = librosa.resample(hrir_r_m80, orig_sr=44100, target_sr=16000)
    hrir_m80_back = np.vstack((hrir_l_m80, hrir_r_m80))
    # -65 deg
    hrir_l_m65 = hrir_l[1,40,:]
    hrir_r_m65 = hrir_r[1,40,:]
    hrir_l_m65 = librosa.resample(hrir_l_m65, orig_sr=44100, target_sr=16000)
    hrir_r_m65 = librosa.resample(hrir_r_m65, orig_sr=44100, target_sr=16000)
    hrir_m65_back = np.vstack((hrir_l_m65, hrir_r_m65))
    # -55 deg
    hrir_l_m55 = hrir_l[2,40,:]
    hrir_r_m55 = hrir_r[2,40,:]
    hrir_l_m55 = librosa.resample(hrir_l_m55, orig_sr=44100, target_sr=16000)
    hrir_r_m55 = librosa.resample(hrir_r_m55,orig_sr= 44100, target_sr=16000)
    hrir_m55_back = np.vstack((hrir_l_m55, hrir_r_m55))
    # -45 deg
    hrir_l_m45 = hrir_l[3,40,:]
    hrir_r_m45 = hrir_r[3,40,:]
    hrir_l_m45 = librosa.resample(hrir_l_m45, orig_sr=44100, target_sr=16000)
    hrir_r_m45 = librosa.resample(hrir_r_m45, orig_sr=44100, target_sr=16000)
    hrir_m45_back = np.vstack((hrir_l_m45, hrir_r_m45))
    # -40 deg
    hrir_l_m40 = hrir_l[4,40,:]
    hrir_r_m40 = hrir_r[4,40,:]
    hrir_l_m40 = librosa.resample(hrir_l_m40, orig_sr=44100, target_sr=16000)
    hrir_r_m40 = librosa.resample(hrir_r_m40, orig_sr=44100, target_sr=16000)
    hrir_m40_back = np.vstack((hrir_l_m40, hrir_r_m40))
    # -35 deg
    hrir_l_m35 = hrir_l[5,40,:]
    hrir_r_m35 = hrir_r[5,40,:]
    hrir_l_m35 = librosa.resample(hrir_l_m35, orig_sr=44100, target_sr=16000)
    hrir_r_m35 = librosa.resample(hrir_r_m35, orig_sr=44100, target_sr=16000)
    hrir_m35_back = np.vstack((hrir_l_m35, hrir_r_m35))
    # -30 deg
    hrir_l_m30 = hrir_l[6,40,:]
    hrir_r_m30 = hrir_r[6,40,:]
    hrir_l_m30 = librosa.resample(hrir_l_m30, orig_sr=44100, target_sr=16000)
    hrir_r_m30 = librosa.resample(hrir_r_m30, orig_sr=44100, target_sr=16000)
    hrir_m30_back = np.vstack((hrir_l_m30, hrir_r_m30))
    # -25 deg
    hrir_l_m25 = hrir_l[7,40,:]
    hrir_r_m25 = hrir_r[7,40,:]
    hrir_l_m25 = librosa.resample(hrir_l_m25, orig_sr=44100, target_sr=16000)
    hrir_r_m25 = librosa.resample(hrir_r_m25, orig_sr=44100, target_sr=16000)
    hrir_m25_back = np.vstack((hrir_l_m25, hrir_r_m25))
    # -20 deg
    hrir_l_m20 = hrir_l[8,40,:]
    hrir_r_m20 = hrir_r[8,40,:]
    hrir_l_m20 = librosa.resample(hrir_l_m20, orig_sr=44100, target_sr=16000)
    hrir_r_m20 = librosa.resample(hrir_r_m20, orig_sr=44100, target_sr=16000)
    hrir_m20_back = np.vstack((hrir_l_m20, hrir_r_m20))
    # -15 deg
    hrir_l_m15 = hrir_l[9,40,:]
    hrir_r_m15 = hrir_r[9,40,:]
    hrir_l_m15 = librosa.resample(hrir_l_m15, orig_sr=44100, target_sr=16000)
    hrir_r_m15 = librosa.resample(hrir_r_m15, orig_sr=44100, target_sr=16000)
    hrir_m15_back = np.vstack((hrir_l_m15, hrir_r_m15))
    # -10 deg
    hrir_l_m10 = hrir_l[10,40,:]
    hrir_r_m10 = hrir_r[10,40,:]
    hrir_l_m10 = librosa.resample(hrir_l_m10, orig_sr=44100, target_sr=16000)
    hrir_r_m10 = librosa.resample(hrir_r_m10, orig_sr=44100, target_sr=16000)
    hrir_m10_back = np.vstack((hrir_l_m10, hrir_r_m10))
    # -5 deg
    hrir_l_m5 = hrir_l[11,40,:]
    hrir_r_m5 = hrir_r[11,40,:]
    hrir_l_m5 = librosa.resample(hrir_l_m5, orig_sr=44100, target_sr=16000)
    hrir_r_m5 = librosa.resample(hrir_r_m5, orig_sr=44100, target_sr=16000)
    hrir_m5_back = np.vstack((hrir_l_m5, hrir_r_m5))
    # 0 deg 
    hrir_l_0 = hrir_l[12,40,:]
    hrir_r_0 = hrir_r[12,40,:]
    hrir_l_0 = librosa.resample(hrir_l_0, orig_sr=44100, target_sr=16000)
    hrir_r_0 = librosa.resample(hrir_r_0, orig_sr=44100, target_sr=16000)
    hrir_0_back = np.vstack((hrir_l_0, hrir_r_0))
    # 5 deg 
    hrir_l_5 = hrir_l[13,40,:]
    hrir_r_5 = hrir_r[13,40,:]
    hrir_l_5 = librosa.resample(hrir_l_5, orig_sr=44100, target_sr=16000)
    hrir_r_5 = librosa.resample(hrir_r_5, orig_sr=44100, target_sr=16000)
    hrir_5_back = np.vstack((hrir_l_5, hrir_r_5))
    # 10 deg 
    hrir_l_10 = hrir_l[14,40,:]
    hrir_r_10 = hrir_r[14,40,:]
    hrir_l_10 = librosa.resample(hrir_l_10, orig_sr=44100, target_sr=16000)
    hrir_r_10 = librosa.resample(hrir_r_10, orig_sr=44100, target_sr=16000)
    hrir_10_back = np.vstack((hrir_l_10, hrir_r_10))
    # 15 deg 
    hrir_l_15 = hrir_l[15,40,:]
    hrir_r_15 = hrir_r[15,40,:]
    hrir_l_15 = librosa.resample(hrir_l_15, orig_sr=44100, target_sr=16000)
    hrir_r_15 = librosa.resample(hrir_r_15, orig_sr=44100, target_sr=16000)
    hrir_15_back = np.vstack((hrir_l_15, hrir_r_15))
    # 20 deg 
    hrir_l_20 = hrir_l[16,40,:]
    hrir_r_20 = hrir_r[16,40,:]
    hrir_l_20 = librosa.resample(hrir_l_20, orig_sr=44100, target_sr=16000)
    hrir_r_20 = librosa.resample(hrir_r_20, orig_sr=44100, target_sr=16000)
    hrir_20_back = np.vstack((hrir_l_20, hrir_r_20))
    # 25 deg 
    hrir_l_25 = hrir_l[17,40,:]
    hrir_r_25 = hrir_r[17,40,:]
    hrir_l_25 = librosa.resample(hrir_l_25, orig_sr=44100, target_sr=16000)
    hrir_r_25 = librosa.resample(hrir_r_25, orig_sr=44100, target_sr=16000)
    hrir_25_back = np.vstack((hrir_l_25, hrir_r_25))
    # 30 deg 
    hrir_l_30 = hrir_l[18,40,:]
    hrir_r_30 = hrir_r[18,40,:]
    hrir_l_30 = librosa.resample(hrir_l_30, orig_sr=44100, target_sr=16000)
    hrir_r_30 = librosa.resample(hrir_r_30, orig_sr=44100, target_sr=16000)
    hrir_30_back = np.vstack((hrir_l_30, hrir_r_30))
    # 35 deg 
    hrir_l_35 = hrir_l[19,40,:]
    hrir_r_35 = hrir_r[19,40,:]
    hrir_l_35 = librosa.resample(hrir_l_35, orig_sr=44100, target_sr=16000)
    hrir_r_35 = librosa.resample(hrir_r_35, orig_sr=44100, target_sr=16000)
    hrir_35_back = np.vstack((hrir_l_35, hrir_r_35))
    # 40 deg 
    hrir_l_40 = hrir_l[20,40,:]
    hrir_r_40 = hrir_r[20,40,:]
    hrir_l_40 = librosa.resample(hrir_l_40, orig_sr=44100, target_sr=16000)
    hrir_r_40 = librosa.resample(hrir_r_40, orig_sr=44100, target_sr=16000)
    hrir_40_back = np.vstack((hrir_l_40, hrir_r_40))
    # 45 deg 
    hrir_l_45 = hrir_l[21,40,:]
    hrir_r_45 = hrir_r[21,40,:]
    hrir_l_45 = librosa.resample(hrir_l_45, orig_sr=44100, target_sr=16000)
    hrir_r_45 = librosa.resample(hrir_r_45, orig_sr=44100, target_sr=16000)
    hrir_45_back = np.vstack((hrir_l_45, hrir_r_45))
    # 55 deg 
    hrir_l_55 = hrir_l[22,40,:]
    hrir_r_55 = hrir_r[22,40,:]
    hrir_l_55 = librosa.resample(hrir_l_55, orig_sr=44100, target_sr=16000)
    hrir_r_55 = librosa.resample(hrir_r_55, orig_sr=44100, target_sr=16000)
    hrir_55_back = np.vstack((hrir_l_55, hrir_r_55))
    # 65 deg 
    hrir_l_65 = hrir_l[23,40,:]
    hrir_r_65 = hrir_r[23,40,:]
    hrir_l_65  = librosa.resample(hrir_l_65 , orig_sr=44100, target_sr=16000)
    hrir_r_65  = librosa.resample(hrir_r_65 , orig_sr=44100, target_sr=16000)
    hrir_65_back = np.vstack((hrir_l_65, hrir_r_65))
    # 80 deg
    hrir_l_80 = hrir_l[24,40,:]
    hrir_r_80 = hrir_r[24,40,:]
    hrir_l_80 = librosa.resample(hrir_l_80, orig_sr=44100, target_sr=16000)
    hrir_r_80 = librosa.resample(hrir_r_80, orig_sr=44100, target_sr=16000)
    hrir_80_back = np.vstack((hrir_l_80, hrir_r_80))

    #all_hrir = [hrir_m80,hrir_m65,hrir_m55,hrir_m45,hrir_m40,hrir_m35,hrir_m30,hrir_m25,hrir_m20,hrir_m15,hrir_m10,hrir_m5,hrir_0,hrir_5,hrir_10,hrir_15,hrir_20,hrir_25,hrir_30, hrir_35, hrir_40, hrir_45, hrir_55, hrir_65, hrir_80, hrir_80_back, hrir_65_back, hrir_55_back, hrir_45_back, hrir_40_back, hrir_35_back, hrir_30_back, hrir_25_back, hrir_20_back, hrir_15_back, hrir_10_back, hrir_5_back, hrir_0_back, hrir_m5_back, hrir_m10_back, hrir_m15_back, hrir_m20_back, hrir_m25_back, hrir_m30_back, hrir_m35_back, hrir_m40_back, hrir_m45_back, hrir_m55_back, hrir_m65_back, hrir_m80_back ]
    
    # 4-point system 
    #all_hrir = [hrir_m80, hrir_0, hrir_80_back, hrir_0_back]

    # 16-point system 
    all_hrir = [hrir_m80, hrir_m65, hrir_m45, hrir_m20, hrir_0, hrir_20, hrir_45, hrir_65, hrir_80_back, hrir_65_back, hrir_45_back, hrir_20_back, hrir_0_back, hrir_m20_back, hrir_m45_back, hrir_m65_back]
    
    # 32-point system 
    #all_hrir = [hrir_m80, hrir_m65, hrir_m45, hrir_m20, hrir_0, hrir_20, hrir_45, hrir_65, hrir_80_back, hrir_65_back, hrir_45_back, hrir_20_back, hrir_0_back, hrir_m20_back, hrir_m45_back, hrir_m65_back]

    #all_hrir = np.array(all_hrir)
    hrir_matrix = np.dstack(all_hrir)



    return all_hrir

all_hrir = extract_hrir(hrir_l_data, hrir_r_data) # n_hrir, n-channels, n-samples (50, 2, 200)

hrir_list = all_hrir

print(len(all_hrir))


# %% CONVOLVE TRAINING

# Convolve training

all_hrir= np.array(all_hrir)

label_list_training = []
data_list_training_clean = []
data_list_training_noisy = []
rand_list = np.linspace(0, len(hrir_list)-1, len(hrir_list))
rand_list = rand_list.astype(int)

degree_list = [10, 25, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 145, 155, 170, 190, 205, 215, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 325, 335, 350]
degree_list = np.array(degree_list)


input_folder_clean = r"D:\Speciale\data\Train - Clean speech audio wav split"
input_folder_noise = r"D:\Speciale\data\Train - Noisy speech audio wav split"

for filename in os.listdir(input_folder_clean):
    input_file_clean = os.path.join(input_folder_clean, filename)
    input_file_noise = os.path.join(input_folder_noise, filename)

    # read file 

    y_clean, fs = sf.read(input_file_clean)
    y_noise, fs = sf.read(input_file_noise)

    noise = np.random.normal(0, 1, 16000*2)

    rand_number = random.choice(rand_list.tolist())

    label_list_vector = np.zeros(len(rand_list))
    label_list_vector[rand_number] = 1
    label_list_training.append(label_list_vector)

    # hrtf_L = hrir_list[rand_number,0,:] # 50,2,200
    # hrtf_R = hrir_list[rand_number,1,:]

    hrtf_L = hrir_list[rand_number][0][:] # 50,2,200
    hrtf_R = hrir_list[rand_number][1][:]

    y_fil_L_clean = np.convolve(y_clean, hrtf_L)
    y_fil_R_clean = np.convolve(y_clean, hrtf_R)

    y_fil_L_noisy = np.convolve(y_noise, hrtf_L)
    y_fil_R_noisy = np.convolve(y_noise, hrtf_R)

    # y_fil_L_clean = np.convolve(noise, hrtf_L)
    # y_fil_R_clean = np.convolve(noise, hrtf_R)

    # y_fil_L_noisy = np.convolve(noise, hrtf_L)
    # y_fil_R_noisy = np.convolve(noise, hrtf_R)


    # STFT
    nperseg = 512
    noverlap = nperseg/2
    f_stft_L, t_stft_L, Zxx_L_clean = stft(y_fil_L_clean , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R_clean = stft(y_fil_R_clean , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    f_stft_L, t_stft_L, Zxx_L_noisy = stft(y_fil_L_noisy , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R_noisy = stft(y_fil_R_noisy , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    stack = np.hstack((Zxx_L_clean, Zxx_R_clean))
    stack = abs(stack)
    #stack = abs(Zxx_L_clean) - abs(Zxx_R_clean)
    stack[stack == 0] = 0.1 # Undgå at dividere med 0 
    stack = 10 * np.log10(stack)

    scale_factor = 90
    stack = stack/scale_factor
    stack = stack +1

    # here
    # n_time = len(stack[0][:])
    # zero_array = np.zeros(n_time)
    # stack = np.vstack((zero_array, stack))
    # label = np.argmax(label_list_vector)
    # stack[0,label] = 1


    data_list_training_clean.append(stack)

    stack = np.hstack((Zxx_L_noisy, Zxx_R_noisy))
    stack = abs(stack)
    stack = 10 * np.log10(stack)
    #stack = abs(Zxx_L_noisy) - abs(Zxx_R_noisy)

    scale_factor = 90
    stack = stack/scale_factor
    stack = stack +1

    # n_time = len(stack[0][:])
    # zero_array = np.zeros(n_time)
    # stack = np.vstack((zero_array, stack))

    data_list_training_noisy.append(stack)

    #stack = abs(Zxx_L) - abs(Zxx_R)
    #stack = abs(Zxx_L) - abs(Zxx_R)
    #stack = Zxx_L - Zxx_R
    #stack = abs(Zxx_L - Zxx_R)

    # Save in matrix

# 4 point w label
np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_training_clean_4directions", data_list_training_clean)
np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_training_noisy_4directions", data_list_training_noisy)
np.save(r"D:\Speciale\data\Convolved data\\" + "Labels_training_4directions", label_list_training)

# 16 point
#np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_training_clean_alldirections", data_list_training_clean)
#np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_training_noisy_alldirections", data_list_training_noisy)
#np.save(r"D:\Speciale\data\Convolved data\\" + "Labels_training_alldirections", label_list_training)




# %% CONVOLVE TESTING


# Convolve test 

label_list_test = []
data_list_test_clean = []
data_list_test_noisy = []

input_folder_clean = r"D:\Speciale\data\Test - Clean speech audio wav split"
input_folder_noise = r"D:\Speciale\data\Test - Noisy speech audio wav split"

for filename in os.listdir(input_folder_clean):
    input_file_clean = os.path.join(input_folder_clean, filename)
    input_file_noise = os.path.join(input_folder_noise, filename)

    # read file 

    y_clean, fs = sf.read(input_file_clean)
    y_noise, fs = sf.read(input_file_noise)

    noise = np.random.normal(0, 1, 16000*2)

    rand_number = random.choice(rand_list.tolist())

    label_list_vector = np.zeros(len(rand_list))
    label_list_vector[rand_number] = 1
    label_list_test.append(label_list_vector)

    # hrtf_L = hrir_list[rand_number,0,:] # 50,2,200
    # hrtf_R = hrir_list[rand_number,1,:]

    hrtf_L = hrir_list[rand_number][0][:] # 50,2,200
    hrtf_R = hrir_list[rand_number][1][:]

    y_fil_L_clean = np.convolve(y_clean, hrtf_L)
    y_fil_R_clean = np.convolve(y_clean, hrtf_R)

    y_fil_L_noisy = np.convolve(y_noise, hrtf_L)
    y_fil_R_noisy = np.convolve(y_noise, hrtf_R)

    # y_fil_L_clean = np.convolve(noise, hrtf_L)
    # y_fil_R_clean = np.convolve(noise, hrtf_R)

    # y_fil_L_noisy = np.convolve(noise, hrtf_L)
    # y_fil_R_noisy = np.convolve(noise, hrtf_R)


    # STFT
    nperseg = 512
    noverlap = nperseg/2
    f_stft_L, t_stft_L, Zxx_L_clean = stft(y_fil_L_clean , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R_clean = stft(y_fil_R_clean , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    f_stft_L, t_stft_L, Zxx_L_noisy = stft(y_fil_L_noisy , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R_noisy = stft(y_fil_R_noisy , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    stack = np.hstack((Zxx_L_clean, Zxx_R_clean))
    stack = abs(stack)
    #stack = abs(Zxx_L_clean) - abs(Zxx_R_clean)
    stack = 10 * np.log10(stack)

    scale_factor = 90
    stack = stack/scale_factor
    stack = stack +1

    # n_time = len(stack[0][:])
    # zero_array = np.zeros(n_time)
    # stack = np.vstack((zero_array, stack))
    # label = np.argmax(label_list_vector)
    # stack[0,label] = 1




    data_list_test_clean.append(stack)

    stack = np.hstack((Zxx_L_noisy, Zxx_R_noisy))
    stack = abs(stack)
    #stack = abs(Zxx_L_noisy) - abs(Zxx_R_noisy)
    stack = 10 * np.log10(stack)

    scale_factor = 90
    stack = stack/scale_factor
    stack = stack +1

    # n_time = len(stack[0][:])
    # zero_array = np.zeros(n_time)
    # stack = np.vstack((zero_array, stack))



    data_list_test_noisy.append(stack)

    #stack = abs(Zxx_L) - abs(Zxx_R)
    #stack = Zxx_L - Zxx_R
    #stack = abs(Zxx_L - Zxx_R)

    # Save in matrix

# 4 point w label
# np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_test_clean_4directions", data_list_test_clean)
# np.save(r"D:\Speciale\data\Convolved data\\" + "Convolve_test_noisy_4directions", data_list_test_noisy)
# np.save(r"D:\Speciale\data\Convolved data\\" + "Labels_test_4directions", label_list_test)






# %% PLOT SANITY CHECK

# PLOT SANITY CHECK

ang = 0

print(np.argmax(label_list_training[ang]))

# Training clean 
plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(data_list_training_clean[ang])
plt.colorbar()

#ang = 44

print(np.argmax(label_list_training[ang]))

# Training clean 
plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(data_list_training_noisy[ang])
plt.colorbar()


# %%

np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\data_list_training_noisy.npy", data_list_training_noisy)
np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\data_list_training_clean.npy", data_list_training_clean)
np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\data_list_test_noisy.npy", data_list_test_noisy)
np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\data_list_test_clean.npy", data_list_test_clean)
np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\label_list_training.npy", label_list_training)
np.save( r"D:\Speciale\data\Convolved data\Convolved data 2" + "\label_list_test.npy", label_list_test)



# # %%

# data_list_training_noisy = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\data_list_training_noisy.npy")
# data_list_training_clean = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\data_list_training_clean.npy")
# data_list_test_noisy = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\data_list_test_noisy.npy")
# data_list_test_clean = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\data_list_test_clean.npy")
# label_list_training = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\label_list_training.npy")
# label_list_test = np.load(r"D:\Speciale\data\Convolved data\Convolved data 2\label_list_test.npy")


# %%

# COMBINE COMBINE COMBINE 

# Regression 1 

#X_train = np.array(data_list_training_noisy)
#Y_train = np.array(data_list_training_clean)
#X_test = np.array(data_list_test_noisy)

X_train = data_list_training_noisy
Y_train = data_list_training_clean
X_test = data_list_test_noisy

X_train_short = X_train[0:1000]
X_test_short = X_test[0:1000]




# %%


# REG & CLASS DER VIRKER! 

rpy.set_seed(42)
rpy.verbosity(0)

# Definer reservoir 
reservoir = Reservoir(1000, lr=0.5, sr=0.9)
ridge_reg = Ridge(ridge=1e-7)
ridge_class = Ridge(ridge=1e-6)

# Gem states 
train_states_reg = []
train_states_class = []

t=time.time()
for x in X_train:
    states = reservoir.run(x, reset=True)

    train_states_reg.append(states)
    train_states_class.append(states[-1, np.newaxis])

readout_reg = ridge_reg.fit(train_states_reg, Y_train, warmup = 10)
ridge_class.fit(train_states_class, label_list_training)
train_time_combi = time.time()-t


test_states = []
Y_pred_reg = []
Y_pred_class = []

t=time.time()
for x in X_test:
    states = reservoir.run(x, reset=True)

    Y_pred_n_reg = readout_reg.run(states)
    Y_pred_n_class = ridge_class.run(states[-1, np.newaxis])

    Y_pred_reg.append(Y_pred_n_reg)
    Y_pred_class.append(Y_pred_n_class)
test_time_combi = time.time()-t

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred_class]
Y_test_class = [np.argmax(y_t) for y_t in label_list_test]

score = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")



#%%
pic = 2

print(np.argmax(label_list_training[pic]))

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(Y_pred_reg[pic], vmin=0, vmax = 1)
plt.colorbar()

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(X_test[pic], vmin=0, vmax = 1)
plt.colorbar()


plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(data_list_test_clean[pic], vmin=0, vmax = 1)
plt.colorbar()

# %%


np.save(r"D:\Speciale\Combi results 2" + "\Y_pred", Y_pred_reg)
np.save(r"D:\Speciale\Combi results 2" + "\X_test", X_test)
np.save(r"D:\Speciale\Combi results 2" + "\data_list_test_clean", data_list_test_clean)
np.save(r"D:\Speciale\Combi results 2" + "\Y_pred_class", Y_pred_class)
np.save(r"D:\Speciale\Combi results 2" + "\Y_test_class", Y_test_class)
np.save(r"D:\Speciale\Combi results 2" + "\Class_score", score)
np.save(r"D:\Speciale\Combi results 2" + "\\train_time_combi", train_time_combi)
np.save(r"D:\Speciale\Combi results 2" + "\\test_time_combi", test_time_combi)




# %%  

# REGRESSION ONLY 

X_train = data_list_training_noisy
Y_train = data_list_training_clean
X_test = data_list_test_noisy

rpy.verbosity(0)  # no need to be too verbose here
rpy.set_seed(42)  # make everything reproducible!

reservoir = Reservoir(1000, lr=0.5, sr=0.9)
ridge = Ridge(ridge=1e-7)

esn_model = reservoir >> ridge

t=time.time()
esn_model = esn_model.fit(X_train, Y_train, warmup=10)
train_time_reg = time.time()-t


print(reservoir.is_initialized, ridge.is_initialized, ridge.fitted)

t=time.time()
Y_pred_reg = esn_model.run(X_test)
test_time_reg = time.time()-t

# %%

# SAVE REG ONLY 
np.save(r"D:\Speciale\Reg only results 2" + "\\train_time_reg", train_time_reg)
np.save(r"D:\Speciale\Reg only results 2" + "\\test_time_reg.npy", test_time_reg)
np.save(r"D:\Speciale\Reg only results 2" + "\Y_pred_reg.npy", Y_pred_reg)
np.save(r"D:\Speciale\Reg only results 2" + "\\data_list_test_clean", data_list_test_clean)
np.save(r"D:\Speciale\Reg only results 2" + "\X_test", X_test)

# %%

Y_pred = np.load(r"D:\Speciale\Combi results\Y_pred.npy")

pic = 0

print(np.argmax(label_list_training[pic]))

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(Y_pred_reg[pic], vmin=0, vmax = 1)
plt.colorbar()

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(Y_pred[pic], vmin=0, vmax = 1)
plt.colorbar()

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(X_test[pic], vmin=0, vmax = 1)
plt.colorbar()

plt.figure()
#plt.pcolor(stack, vmin=-0.1, vmax=0.7)
plt.pcolor(data_list_test_clean[pic], vmin=0, vmax = 1)
plt.colorbar()

# %%

# Load data 

# np.save(r"D:\Speciale\Combi results 2" + "\Y_pred", Y_pred_reg)
# np.save(r"D:\Speciale\Combi results 2" + "\X_test", X_test)
# np.save(r"D:\Speciale\Combi results 2" + "\data_list_test_clean", data_list_test_clean)
# np.save(r"D:\Speciale\Combi results 2" + "\Y_pred_class", Y_pred_class)
# np.save(r"D:\Speciale\Combi results 2" + "\Y_test_class", Y_test_class)
# np.save(r"D:\Speciale\Combi results 2" + "\Class_score", score)
# np.save(r"D:\Speciale\Combi results 2" + "\\train_time_combi", train_time_combi)
# np.save(r"D:\Speciale\Combi results 2" + "\\test_time_combi", test_time_combi)




# %%
# CLASSIFICATION ONLY 


rpy.set_seed(42)
rpy.verbosity(0)

reservoir = Reservoir(1000, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

Y_train = label_list_training

model = reservoir >> readout

states_train = []

t=time.time()
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])

readout.fit(states_train, Y_train)
train_time_class = time.time()-t

Y_pred_class = []

t=time.time()
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states[-1, np.newaxis])
    Y_pred_class.append(y)
test_time_class = time.time()-t

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred_class]
Y_test_class = [np.argmax(y_t) for y_t in label_list_test]

score_class = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score * 100:.3f} %")

# %%

# SAVE CLASS ONLY 
np.save(r"D:\Speciale\Class only results 2" + "\\train_time_class.npy", train_time_class)
np.save(r"D:\Speciale\Class only results 2" + "\\test_time_class.npy", test_time_class)
np.save(r"D:\Speciale\Class only results 2" + "\Y_pred_class.npy", Y_pred_class)
np.save(r"D:\Speciale\Class only results 2" + "\Y_test_class.npy", Y_test_class)
np.save(r"D:\Speciale\Class only results 2" + "\X_train.npy", X_train)
np.save(r"D:\Speciale\Class only results 2" + "\X_test.npy", X_test)
np.save(r"D:\Speciale\Class only results 2" + "\Y_train.npy", label_list_training)
np.save(r"D:\Speciale\Class only results 2" + "\Y_test.npy", label_list_test)
np.save(r"D:\Speciale\Class only results 2" + "\score_class.npy", score_class)
# %%


# CNN RE DO 


import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 260, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10))
model.add(layers.Dense(500)) # 50 classes

model.summary()


# Adam is the best among the adaptive optimizers in most of the cases
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


 # %%


label_list_CNN = []
for i in range(len(label_list_training)):
    label_list_CNN_max = np.argmax(label_list_training[i])
    #print(label_list_CNN_max)
    label_list_CNN.append(label_list_CNN_max)


label_list_test_CNN = []
for i in range(len(label_list_test)):
    label_list_test_CNN_max = np.argmax(label_list_test[i])
    #print(label_list_CNN_max)
    label_list_test_CNN.append(label_list_test_CNN_max)



# %%

train_images = data_list_training_clean
train_labels = label_list_CNN

test_images = data_list_test_clean
test_labels = label_list_test_CNN

train_images = np.array(train_images)
test_images = np.array(test_images)

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

train_images = np.delete(train_images, 0, axis = 1)
#train_images = np.delete(train_images, -1, axis = 2)

test_images = np.delete(test_images, 0, axis = 1)
#test_images = np.delete(test_images, -1, axis = 2)

#train_images = np.reshape(train_images, (5000, 257, 357, 1))
#test_images = np.reshape(test_images, (257, 357, 1))

train_labels = np.array(train_labels)
test_labels = np.array(test_labels)


# %%DATA SPLIT

X_train, X_val, Y_train, Y_val = train_test_split(train_images, train_labels, test_size=0.15)

train_images = X_train
train_labels = Y_train
val_images = X_val 
val_labels = Y_val


# %%

# Training CNN

# An epoch means training the neural network with all the
# training data for one cycle. Here I use 10 epochs

t=time.time()
history = model.fit(train_images, train_labels, epochs=2, 
                    validation_data=(val_images, val_labels))
training_time_CNN = time.time()-t

# %%

# Testing CNN

plt.plot(history.history['accuracy'],label='accuracy')
plt.plot(history.history['val_accuracy'],label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# test_loss, test_acc = model.evaluate(test_images,
#                                      test_labels,
#                                      verbose=2)

#print('Test Accuracy is',test_acc)

t=time.time()
Y_pred = model.predict(test_images, batch_size=16)
testing_time_CNN = time.time()-t

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
#Y_test_class = [np.argmax(y_t) for y_t in test_labels]

score_cnn = accuracy_score(test_labels, Y_pred_class)

print("Accuracy: ", f"{score_cnn * 100:.3f} %")

# %%

# SAVE CNN CLASS REDO

path = r"D:\Speciale\Speech SSL results\\"
np.save(path + "score_cnn_alldirections.npy", score_cnn)
np.save(path + "time_training_cnn_test_alldirections.npy", training_time_CNN)
np.save(path + "time_test_cnn_test_alldirections.npy", testing_time_CNN)
np.save(path + "Y_pred_class_cnn_test_alldirections.npy", Y_pred_class)
np.save(path + "Y_test_class_cnn_test_alldirections.npy", test_labels)
# %%

# Inspect results / F1 

score_4_cnn = np.load(path + "score_cnn_4directions.npy") 
score_16_cnn = np.load(path + "score_cnn_16directions.npy") 
score_all_cnn = np.load(path + "score_cnn_alldirections.npy") 

score_4_esn = np.load(path + "score_esn_4directions.npy") 
score_16_esn = np.load(path + "score_esn_16directions.npy") 
score_all_esn = np.load(path + "score_esn_alldirections.npy") 

# %%

# F1 
import sklearn

f1 = sklearn.metrics.f1_score(test_labels, Y_pred_class, labels=None, pos_label=1, average='weighted', sample_weight=None)
