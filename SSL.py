# %% 

# PIPIC IMPORT 

import scipy.io
import numpy as np 
import matplotlib.pyplot as plt
from scipy.signal import stft, istft
import random 
from collections import defaultdict
from reservoirpy.datasets import japanese_vowels
from reservoirpy import set_seed, verbosity
from reservoirpy.observables import nrmse, rsquare
from reservoirpy.nodes import Reservoir, Ridge, Input
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn import metrics
import os
import re
import time
import scipy.io as sio

from sklearn.model_selection import train_test_split

# Import structure containing HRIR
path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Classification environment\HRTF\cipic-hrtf-database-master\standard_hrir_database\subject_003\hrir_final.mat"

fs = 44100

# Load the .mat file
data = scipy.io.loadmat(path)
hrir_l_data = data["hrir_l"]
hrir_r_data = data["hrir_r"]

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
    hrir_m80 = np.vstack((hrir_l_m80, hrir_r_m80))
    # -65 deg
    hrir_l_m65 = hrir_l[1,8,:]
    hrir_r_m65 = hrir_r[1,8,:]
    hrir_m65 = np.vstack((hrir_l_m65, hrir_r_m65))
    # -55 deg
    hrir_l_m55 = hrir_l[2,8,:]
    hrir_r_m55 = hrir_r[2,8,:]
    hrir_m55 = np.vstack((hrir_l_m55, hrir_r_m55))
    # -45 deg
    hrir_l_m45 = hrir_l[3,8,:]
    hrir_r_m45 = hrir_r[3,8,:]
    hrir_m45 = np.vstack((hrir_l_m45, hrir_r_m45))
    # -40 deg
    hrir_l_m40 = hrir_l[4,8,:]
    hrir_r_m40 = hrir_r[4,8,:]
    hrir_m40 = np.vstack((hrir_l_m40, hrir_r_m40))
    # -35 deg
    hrir_l_m35 = hrir_l[5,8,:]
    hrir_r_m35 = hrir_r[5,8,:]
    hrir_m35 = np.vstack((hrir_l_m35, hrir_r_m35))
    # -30 deg
    hrir_l_m30 = hrir_l[6,8,:]
    hrir_r_m30 = hrir_r[6,8,:]
    hrir_m30 = np.vstack((hrir_l_m30, hrir_r_m30))
    # -25 deg
    hrir_l_m25 = hrir_l[7,8,:]
    hrir_r_m25 = hrir_r[7,8,:]
    hrir_m25 = np.vstack((hrir_l_m25, hrir_r_m25))
    # -20 deg
    hrir_l_m20 = hrir_l[8,8,:]
    hrir_r_m20 = hrir_r[8,8,:]
    hrir_m20 = np.vstack((hrir_l_m20, hrir_r_m20))
    # -15 deg
    hrir_l_m15 = hrir_l[9,8,:]
    hrir_r_m15 = hrir_r[9,8,:]
    hrir_m15 = np.vstack((hrir_l_m15, hrir_r_m15))
    # -10 deg
    hrir_l_m10 = hrir_l[10,8,:]
    hrir_r_m10 = hrir_r[10,8,:]
    hrir_m10 = np.vstack((hrir_l_m10, hrir_r_m10))
    # -5 deg
    hrir_l_m5 = hrir_l[11,8,:]
    hrir_r_m5 = hrir_r[11,8,:]
    hrir_m5 = np.vstack((hrir_l_m5, hrir_r_m5))
    # 0 deg 
    hrir_l_0 = hrir_l[12,8,:]
    hrir_r_0 = hrir_r[12,8,:]
    hrir_0 = np.vstack((hrir_l_0, hrir_r_0))
    # 5 deg 
    hrir_l_5 = hrir_l[13,8,:]
    hrir_r_5 = hrir_r[13,8,:]
    hrir_5 = np.vstack((hrir_l_5, hrir_r_5))
    # 10 deg 
    hrir_l_10 = hrir_l[14,8,:]
    hrir_r_10 = hrir_r[14,8,:]
    hrir_10 = np.vstack((hrir_l_10, hrir_r_10))
    # 15 deg 
    hrir_l_15 = hrir_l[15,8,:]
    hrir_r_15 = hrir_r[15,8,:]
    hrir_15 = np.vstack((hrir_l_15, hrir_r_15))
    # 20 deg 
    hrir_l_20 = hrir_l[16,8,:]
    hrir_r_20 = hrir_r[16,8,:]
    hrir_20 = np.vstack((hrir_l_20, hrir_r_20))
    # 25 deg 
    hrir_l_25 = hrir_l[17,8,:]
    hrir_r_25 = hrir_r[17,8,:]
    hrir_25 = np.vstack((hrir_l_25, hrir_r_25))
    # 30 deg 
    hrir_l_30 = hrir_l[18,8,:]
    hrir_r_30 = hrir_r[18,8,:]
    hrir_30 = np.vstack((hrir_l_30, hrir_r_30))
    # 35 deg 
    hrir_l_35 = hrir_l[19,8,:]
    hrir_r_35 = hrir_r[19,8,:]
    hrir_35 = np.vstack((hrir_l_35, hrir_r_35))
    # 40 deg 
    hrir_l_40 = hrir_l[20,8,:]
    hrir_r_40 = hrir_r[20,8,:]
    hrir_40 = np.vstack((hrir_l_40, hrir_r_40))
    # 45 deg 
    hrir_l_45 = hrir_l[21,8,:]
    hrir_r_45 = hrir_r[21,8,:]
    hrir_45 = np.vstack((hrir_l_45, hrir_r_45))
    # 55 deg 
    hrir_l_55 = hrir_l[22,8,:]
    hrir_r_55 = hrir_r[22,8,:]
    hrir_55 = np.vstack((hrir_l_55, hrir_r_55))
    # 65 deg 
    hrir_l_65 = hrir_l[23,8,:]
    hrir_r_65 = hrir_r[23,8,:]
    hrir_65 = np.vstack((hrir_l_65, hrir_r_65))
    # 80 deg
    hrir_l_80 = hrir_l[24,8,:]
    hrir_r_80 = hrir_r[24,8,:]
    hrir_80 = np.vstack((hrir_l_80, hrir_r_80))

    # For back of sphere 

    # -80 deg
    hrir_l_m80 = hrir_l[0,40,:]
    hrir_r_m80 = hrir_r[0,40,:]
    hrir_m80_back = np.vstack((hrir_l_m80, hrir_r_m80))
    # -65 deg
    hrir_l_m65 = hrir_l[1,40,:]
    hrir_r_m65 = hrir_r[1,40,:]
    hrir_m65_back = np.vstack((hrir_l_m65, hrir_r_m65))
    # -55 deg
    hrir_l_m55 = hrir_l[2,40,:]
    hrir_r_m55 = hrir_r[2,40,:]
    hrir_m55_back = np.vstack((hrir_l_m55, hrir_r_m55))
    # -45 deg
    hrir_l_m45 = hrir_l[3,40,:]
    hrir_r_m45 = hrir_r[3,40,:]
    hrir_m45_back = np.vstack((hrir_l_m45, hrir_r_m45))
    # -40 deg
    hrir_l_m40 = hrir_l[4,40,:]
    hrir_r_m40 = hrir_r[4,40,:]
    hrir_m40_back = np.vstack((hrir_l_m40, hrir_r_m40))
    # -35 deg
    hrir_l_m35 = hrir_l[5,40,:]
    hrir_r_m35 = hrir_r[5,40,:]
    hrir_m35_back = np.vstack((hrir_l_m35, hrir_r_m35))
    # -30 deg
    hrir_l_m30 = hrir_l[6,40,:]
    hrir_r_m30 = hrir_r[6,40,:]
    hrir_m30_back = np.vstack((hrir_l_m30, hrir_r_m30))
    # -25 deg
    hrir_l_m25 = hrir_l[7,40,:]
    hrir_r_m25 = hrir_r[7,40,:]
    hrir_m25_back = np.vstack((hrir_l_m25, hrir_r_m25))
    # -20 deg
    hrir_l_m20 = hrir_l[8,40,:]
    hrir_r_m20 = hrir_r[8,40,:]
    hrir_m20_back = np.vstack((hrir_l_m20, hrir_r_m20))
    # -15 deg
    hrir_l_m15 = hrir_l[9,40,:]
    hrir_r_m15 = hrir_r[9,40,:]
    hrir_m15_back = np.vstack((hrir_l_m15, hrir_r_m15))
    # -10 deg
    hrir_l_m10 = hrir_l[10,40,:]
    hrir_r_m10 = hrir_r[10,40,:]
    hrir_m10_back = np.vstack((hrir_l_m10, hrir_r_m10))
    # -5 deg
    hrir_l_m5 = hrir_l[11,40,:]
    hrir_r_m5 = hrir_r[11,40,:]
    hrir_m5_back = np.vstack((hrir_l_m5, hrir_r_m5))
    # 0 deg 
    hrir_l_0 = hrir_l[12,40,:]
    hrir_r_0 = hrir_r[12,40,:]
    hrir_0_back = np.vstack((hrir_l_0, hrir_r_0))
    # 5 deg 
    hrir_l_5 = hrir_l[13,40,:]
    hrir_r_5 = hrir_r[13,40,:]
    hrir_5_back = np.vstack((hrir_l_5, hrir_r_5))
    # 10 deg 
    hrir_l_10 = hrir_l[14,40,:]
    hrir_r_10 = hrir_r[14,40,:]
    hrir_10_back = np.vstack((hrir_l_10, hrir_r_10))
    # 15 deg 
    hrir_l_15 = hrir_l[15,40,:]
    hrir_r_15 = hrir_r[15,40,:]
    hrir_15_back = np.vstack((hrir_l_15, hrir_r_15))
    # 20 deg 
    hrir_l_20 = hrir_l[16,40,:]
    hrir_r_20 = hrir_r[16,40,:]
    hrir_20_back = np.vstack((hrir_l_20, hrir_r_20))
    # 25 deg 
    hrir_l_25 = hrir_l[17,40,:]
    hrir_r_25 = hrir_r[17,40,:]
    hrir_25_back = np.vstack((hrir_l_25, hrir_r_25))
    # 30 deg 
    hrir_l_30 = hrir_l[18,40,:]
    hrir_r_30 = hrir_r[18,40,:]
    hrir_30_back = np.vstack((hrir_l_30, hrir_r_30))
    # 35 deg 
    hrir_l_35 = hrir_l[19,40,:]
    hrir_r_35 = hrir_r[19,40,:]
    hrir_35_back = np.vstack((hrir_l_35, hrir_r_35))
    # 40 deg 
    hrir_l_40 = hrir_l[20,40,:]
    hrir_r_40 = hrir_r[20,40,:]
    hrir_40_back = np.vstack((hrir_l_40, hrir_r_40))
    # 45 deg 
    hrir_l_45 = hrir_l[21,40,:]
    hrir_r_45 = hrir_r[21,40,:]
    hrir_45_back = np.vstack((hrir_l_45, hrir_r_45))
    # 55 deg 
    hrir_l_55 = hrir_l[22,40,:]
    hrir_r_55 = hrir_r[22,40,:]
    hrir_55_back = np.vstack((hrir_l_55, hrir_r_55))
    # 65 deg 
    hrir_l_65 = hrir_l[23,40,:]
    hrir_r_65 = hrir_r[23,40,:]
    hrir_65_back = np.vstack((hrir_l_65, hrir_r_65))
    # 80 deg
    hrir_l_80 = hrir_l[24,40,:]
    hrir_r_80 = hrir_r[24,40,:]
    hrir_80_back = np.vstack((hrir_l_80, hrir_r_80))

    all_hrir = [hrir_m80,hrir_m65,hrir_m55,hrir_m45,hrir_m40,hrir_m35,hrir_m30,hrir_m25,hrir_m20,hrir_m15,hrir_m10,hrir_m5,hrir_0,hrir_5,hrir_10,hrir_15,hrir_20,hrir_25,hrir_30, hrir_35, hrir_40, hrir_45, hrir_55, hrir_65, hrir_80, hrir_80_back, hrir_65_back, hrir_55_back, hrir_45_back, hrir_40_back, hrir_35_back, hrir_30_back, hrir_25_back, hrir_20_back, hrir_15_back, hrir_10_back, hrir_5_back, hrir_0_back, hrir_m5_back, hrir_m10_back, hrir_m15_back, hrir_m20_back, hrir_m25_back, hrir_m30_back, hrir_m35_back, hrir_m40_back, hrir_m45_back, hrir_m55_back, hrir_m65_back, hrir_m80_back ]
    all_hrir = np.array(all_hrir)
    #hrir_matrix = np.dstack(all_hrir)

    return all_hrir


all_hrir = extract_hrir(hrir_l_data, hrir_r_data) # n_hrir, n-channels, n-samples (50, 2, 200)

hrir_d90 = np.array([all_hrir[0], all_hrir[12], all_hrir[25], all_hrir[37]])

hrir_d45 = np.array([all_hrir[0], all_hrir[3], all_hrir[12], all_hrir[21], all_hrir[25], all_hrir[28], all_hrir[37], all_hrir[46]])

hrir_d45 = np.array([all_hrir[0], all_hrir[3], all_hrir[12], all_hrir[21], all_hrir[25], all_hrir[29], all_hrir[37], all_hrir[44]])

hrir_list = all_hrir

#every_x = all_hrir[::2, :, :]
#every_x = all_hrir[::3, :, :]
#every_x = all_hrir[::4, :, :]

#all_hrir = every_x


# %% 

# FUCK AROUND ! BEHOLD! 
for i in range (0,len(hrir_list),1):

    n = i
    hrir_L = hrir_list[n,0,:]
    hrir_R = hrir_list[n,1,:]

    noise = np.random.normal(0, 1, fs*2)

    y_fil_L = np.convolve(noise, hrir_L)
    y_fil_R = np.convolve(noise, hrir_R)

    # STFT
    nperseg = 512
    noverlap = nperseg/2
    f_stft_L, t_stft_L, Zxx_L = stft(y_fil_L , fs=44100, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R = stft(y_fil_R , fs=44100, nperseg=nperseg, noverlap=noverlap, window="hamming")

    stack = np.hstack((Zxx_L, Zxx_R))
    #stack = abs(stack)

    #diff = abs(Zxx_L - Zxx_R)
    diff = abs(Zxx_L) - abs(Zxx_R)
    #diff = Zxx_L - Zxx_R
    print(np.sum(diff))

# # Sanity check 

# #print(label_list_test)
# n = 0

# plt.figure()
# #plt.pcolor(stack, vmin=-0.1, vmax=0.7)
# plt.pcolor(stack)
# plt.colorbar()

# plt.figure()
# #plt.pcolor(stack, vmin=-0.1, vmax=0.7)
# plt.pcolor(diff)
# plt.colorbar()

print(np.sum(diff))

# %% 

#Make data training 

label_list_training = []
data_list_training = []
rand_list = np.linspace(0, len(hrir_list[:,0,0])-1, len(hrir_list[:,0,0]))
rand_list = rand_list.astype(int)

degree_list = [10, 25, 35, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120, 125, 130, 135, 145, 155, 170, 190, 205, 215, 225, 230, 235, 240, 245, 250, 255, 260, 265, 270, 275, 280, 285, 290, 295, 300, 305, 310, 315, 325, 335, 350]
degree_list = np.array(degree_list)


for i in range(5000):

    rand_number = random.choice(rand_list)
    label_list_vector = np.zeros(len(rand_list))
    label_list_vector[rand_number] = 1
    label_list_training.append(label_list_vector)

    hrtf_L = hrir_list[rand_number,0,:] # 50,2,200
    hrtf_R = hrir_list[rand_number,1,:]

    noise = np.random.normal(0, 1, fs*2)

    y_fil_L = np.convolve(noise, hrtf_L)
    y_fil_R = np.convolve(noise, hrtf_R)

    # STFT
    nperseg = 512
    noverlap = nperseg/2
    f_stft_L, t_stft_L, Zxx_L = stft(y_fil_L , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R = stft(y_fil_R , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    #stack = np.hstack((Zxx_L, Zxx_R))
    #stack = abs(stack)
    stack = abs(Zxx_L) - abs(Zxx_R)
    #stack = Zxx_L - Zxx_R
    #stack = abs(Zxx_L - Zxx_R)

    # Save in matrix
    data_list_training.append(stack)

print(np.array(data_list_training).shape)
print(len(label_list_training[0]))


# %% ___________________

#Make data testing 

label_list_test = []
data_list_test = []

for i in range(1000):

    rand_number = random.choice(rand_list)
    label_list_vector = np.zeros(len(rand_list))
    label_list_vector[rand_number] = 1
    label_list_test.append(label_list_vector)

    hrtf_L = hrir_list[rand_number,0,:] # 50,2,200
    hrtf_R = hrir_list[rand_number,1,:]

    noise = np.random.normal(0, 1, fs*2)

    y_fil_L = np.convolve(noise, hrtf_L)
    y_fil_R = np.convolve(noise, hrtf_R)

    # STFT
    nperseg = 512
    noverlap = nperseg/2
    f_stft_L, t_stft_L, Zxx_L = stft(y_fil_L , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")
    f_stft_R, t_stft_R, Zxx_R = stft(y_fil_R , fs=16000, nperseg=nperseg, noverlap=noverlap, window="hamming")

    #tack = np.hstack((Zxx_L, Zxx_R))
    #stack = abs(stack)
    stack = abs(Zxx_L) - abs(Zxx_R)
    #stack = Zxx_L - Zxx_R
    #stack = abs(Zxx_L - Zxx_R)

    data_list_test.append(stack)

print(np.array(data_list_test).shape)
print(len(label_list_test[0]))


# %%___________________

# # Sanity check 

# #print(label_list_test)
# n = 12
# print(label_list_test[n])
# print(np.argmax(label_list_test[n]))
# plt.figure()
# plt.pcolor(data_list_test[n], vmin=-0.1, vmax=0.7)
# #plt.pcolor(data_list_test[n])
# plt.colorbar()

# print(np.sum(data_list_test[n]))

# %%____________________

# Training ESN

X_train = data_list_training
Y_train = label_list_training

set_seed(42)
verbosity(0)

source = Input()
reservoir = Reservoir(1000, sr=0.9, lr=0.1)
readout = Ridge(ridge=1e-6)

model = source >> reservoir >> readout

states_train = []

t=time.time()
for x in X_train:
    states = reservoir.run(x, reset=True)
    states_train.append(states[-1, np.newaxis])


readout.fit(states_train, Y_train)
training_time_esn = time.time()-t



# %%____________________

# Testing ESN

X_test = data_list_test
Y_test = label_list_test

Y_pred = []

t=time.time()
for x in X_test:
    states = reservoir.run(x, reset=True)
    y = readout.run(states[-1, np.newaxis])
    Y_pred.append(y)
testing_time_esn = time.time()-t


# %%______________________

# Results ESN

Y_pred_class = [np.argmax(y_p) for y_p in Y_pred]
Y_test_class = [np.argmax(y_t) for y_t in Y_test]
print(len(np.unique(Y_pred_class)))
print(len(np.unique(Y_test_class)))

Y_pred_class_deg = degree_list[np.array(Y_pred_class)]
Y_test_class_deg = degree_list[np.array(Y_test_class)]

score_esn = accuracy_score(Y_test_class, Y_pred_class)

print("Accuracy: ", f"{score_esn * 100:.3f} %")

print(Y_pred_class)
print(Y_test_class)

print(max(abs(np.array(Y_pred_class) - np.array(Y_test_class))))

np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Classification environment\results\Python export\\" + "Y_pred_class_esn", Y_pred_class_deg)
np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Classification environment\results\Python export\\" + "Y_test_class_esn", Y_test_class_deg)

# %%

# Confusion matrix

confusion_matrix = metrics.confusion_matrix(Y_pred_class, Y_test_class)

#cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)

cm_display.plot()
plt.show()


# %%

# CNN  COPY 


#data_list_training_norm = (np.array(data_list_training +1)) / (np.max(np.array(data_list_training))) 


# Normalize pixel values to be between 0 and 1
#train_images, test_images = train_images / 255.0, test_images / 255.0
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 346, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
#model.add(layers.Dense(10))
model.add(layers.Dense(50)) # 50 classes

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

train_images = data_list_training
train_labels = label_list_CNN

test_images = data_list_test
test_labels = label_list_test_CNN

train_images = np.array(train_images)
test_images = np.array(test_images)

train_images = train_images[..., np.newaxis]
test_images = test_images[..., np.newaxis]

train_images = np.delete(train_images, 0, axis = 1)
train_images = np.delete(train_images, -1, axis = 2)

test_images = np.delete(test_images, 0, axis = 1)
test_images = np.delete(test_images, -1, axis = 2)

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

# Save results

run = "5"

times_matrix = ([training_time_esn, testing_time_esn, training_time_CNN, testing_time_CNN])

path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Classification environment\results\Python export\\"
np.save(path +  "classification times_run_" + run, times_matrix) # train esn, test esn, train cnn, test cnn

score_matrix=([score_esn, score_cnn])
np.save(path + "classification scores_run_" + run, score_matrix)

def convert_npy_to_mat(folder_path):

    # List all files in the folder
    files = os.listdir(folder_path)
    
    for file in files:
        # Check if the file is a .npy file
        if file.endswith('.npy'):
            # Construct the full file path
            npy_file_path = os.path.join(folder_path, file)
            
            # Load the .npy file
            data = np.load(npy_file_path)
            
            # Construct the .mat file name
            mat_file_path = os.path.join(folder_path, file.replace('.npy', '.mat'))
            
            # Save the data to a .mat file
            sio.savemat(mat_file_path, {'data': data})
            
            #print(f"Converted {file} to {os.path.basename(mat_file_path)}")

convert_npy_to_mat(path)

# %%

