#%% ESN/ CNN Regression combined 
os.chdir(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment")

import os
import sys
import importlib
import numpy as np
import functions
import Prepare_data
import ESN
import soundfile as sf
import evaluate
import matplotlib.pyplot as plt
import random as rnd
import time
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dropout, BatchNormalization, Input
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.utils 
import keras
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir
from reservoirpy.nodes import Ridge

importlib.reload(ESN)
importlib.reload(Prepare_data)
importlib.reload(functions)
importlib.reload(evaluate)



# set main path 
path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\\"

# sampling freq 
fs = 16000

# Type of noise 
noise_type = "pink_noise"

# Model type
model_type = "CNN"

snr_list = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]

#snr_list = [-10]

# Noise factor
noise_factor = 0.05

#Segment length in samples  
segment_length = 1024*32 # 1024 er 64 ms 2 SEK  

print("segment length: ", segment_length/fs, "seconds")

# %% PREPARE DATA

for i in range(len(snr_list)):

    snr = snr_list[i]
    
    Prepare_data.make_data(path, noise_factor, segment_length, noise_type, snr)

# X_TEST_CLEAN
INPUT_DIR  = path + r"Test - Clean speech audio wav split"
OUTPUT_DIR = path + r"arrays_NPY_Drive"
output_name_abs = "\\X_Test_Clean_abs_split_" + noise_type 
output_name_phase = "\\X_Test_Clean_phase_split_" + noise_type
X_Test_Clean_abs_split, X_Test_Clean_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)

print(np.array(X_Test_Clean_abs_split).shape)






# %% CNN Model 



# LOOP START

seed_list = [0, 10, 20, 30, 40]
seed_list = [40]

snr_list = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]



for i in range(len(seed_list)):

    seed = seed_list[i]
    tensorflow.keras.utils.set_random_seed(seed)

    # Model type
    model_type = "CNN"

    path = r'C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\arrays_NPY_Drive\\'

    #snr_list = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]
    #snr_list = [10]

    Y_pred_matrix_CNN = np.zeros((7,len(snr_list)))

    #keras.backend.clear_session()

    for i in range(len(snr_list)):

        #keras.backend.clear_session()
        
        snr = snr_list[i]
        #print(i)
        #print(snr)
        # Import data 

        print(path + "X_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")

        X_train_abs          = np.load(path + "/X_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        Y_train_abs          = np.load(path + "/Y_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_abs           = np.load(path + "/X_Test_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_phase         = np.load(path + "/X_Test_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_clean_abs     = np.load(path + "/X_Test_Clean_abs_split_" + noise_type + ".npy")
        X_test_clean_phase   = np.load(path + "/X_Test_Clean_phase_split_" + noise_type + ".npy")

        # SPLIT DATA

        X_train, X_val, Y_train, Y_val = train_test_split(X_train_abs, Y_train_abs, test_size=0.15)

        #  MODELING

        def model():

            input_layer = Input(shape=(256, 128, 1))  # we might define (None,None,1) here, but in model summary dims would not be visible

            # encoding
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(input_layer)
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = MaxPooling2D((2, 2), padding='same')(x)
            x = Dropout(0.5)(x)

            # decoding
            x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
            x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
            x = BatchNormalization()(x)
            x = UpSampling2D((2, 2))(x)

            output_layer = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
            model = Model(inputs=[input_layer], outputs=[output_layer])
            model.compile(optimizer='adam' , loss='mean_squared_error', metrics=['mae'])

            return model


        model = model()
        model.summary()

        # TRAIN MODEL

        #callback = EarlyStopping(monitor='loss', patience=10)
        callback = EarlyStopping(monitor='loss', patience=10, restore_best_weights = True)

        #history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=30, batch_size=6, verbose=1, callbacks=[callback]) # DEN DER VRIKER
        print("beginning traing of " + str(snr) + " dB")

        # Train and save time
        
        t=time.time()
        history = model.fit(X_train, Y_train, validation_data = (X_val, Y_val), epochs=100, batch_size=16, verbose=1, callbacks=[callback])
        train_time = time.time()-t

        np.save(path + "train_time" + model_type +  noise_type + "_" + str(snr) + "_dB.npy", train_time)
        
        #Check how loss & mae went down
        epoch_loss = history.history['loss']
        epoch_val_loss = history.history['val_loss']
        epoch_mae = history.history['mae']
        epoch_val_mae = history.history['val_mae']

        plt.figure(figsize=(20,6))
        plt.subplot(1,2,1)
        plt.plot(range(0,len(epoch_loss)), epoch_loss, 'b-', linewidth=2, label='Train Loss')
        plt.plot(range(0,len(epoch_val_loss)), epoch_val_loss, 'r-', linewidth=2, label='Val Loss')
        plt.title('Evolution of loss on train & validation datasets over epochs')
        plt.legend(loc='best')

        plt.subplot(1,2,2)
        plt.plot(range(0,len(epoch_mae)), epoch_mae, 'b-', linewidth=2, label='Train MAE')
        plt.plot(range(0,len(epoch_val_mae)), epoch_val_mae, 'r-', linewidth=2,label='Val MAE')
        plt.title('Evolution of MAE on train & validation datasets over epochs')
        plt.legend(loc='best')
        plt.show()

        # Test and save time

        t=time.time()
        Y_pred = model.predict(X_test_abs, batch_size=16)
        test_time = time.time()-t

        np.save(path + "test_time" + model_type +  noise_type + "_" + str(snr) + "_dB.npy", test_time)

        #Save model
        model.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Models\CNN_regression_model_003_" + noise_type + "_" + str(snr) + ".keras")  # For Keras models

        print("CNN model done predicting for " + str(snr) + " dB" + "seed is " + str(seed))

        alpha = 0.01
        threshold = 0.02

        X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
        Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
        X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
        X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))
        Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], Y_pred.shape[1], Y_pred.shape[2]))

        [pesq_score_Clean, pesq_score_Test, pesq_score_y_pred, snr_out]  = evaluate.eval2(Y_pred, X_test_phase, X_test_abs, X_test_clean_abs, X_test_clean_phase, alpha, threshold)

        print(str(snr) + " is " + str(snr_out))

        # Save results
        np.save(path + "Y_pred" + model_type + noise_type + "_" + str(snr) + "_dB.npy", Y_pred)
        np.save(path + "train_time" + model_type + noise_type + "_" + str(snr) + "_dB.npy", train_time)
        np.save(path + "test_time" + model_type + noise_type + "_" + str(snr) + "_dB.npy", test_time)


    print(str(snr) +"_done")




    #  EVALUATE CNN MODEL OUTPUT 




    importlib.reload(functions)
    importlib.reload(evaluate)




    path = r'C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\arrays_NPY_Drive\\'

    results_matrix_CNN = np.zeros((7,len(snr_list)))

    # For SNR out calculation

    alpha = 0.01
    threshold = 0.1e8
    threshold = 0.02 # 09/01


    for i in range(len(snr_list)):

        snr = snr_list[i]

        # IMPORT NPY
        X_train_abs, Y_train_abs, X_test_abs, X_test_phase, X_test_clean_abs, X_test_clean_phase = functions.import_NPY_no_phase(path, snr, noise_type)

        Y_pred = np.load(path  + "Y_pred" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

        # Reshape 
        X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
        Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
        X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
        X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))
        Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], Y_pred.shape[1], Y_pred.shape[2]))

        # % POST PROCESS TO AUDIO
        output_length = segment_length/16000 
        input_length = segment_length
        output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\processed audio data"
        test_label = "_" + model_type + "_" + noise_type + "_" + str(snr) + "_" + r".wav"
        
        #Y_pred
        functions.concatenate_signal(Y_pred, X_test_phase, input_length, output_length, output_dir, output_name = r"\CNN_model_output" + test_label)
        # X_test
        functions.concatenate_signal(X_test_abs, X_test_phase, input_length, output_length, output_dir, output_name = r"\CNN_X_Test" + test_label)
        # X_test_clean
        functions.concatenate_signal(X_test_clean_abs, X_test_clean_phase, input_length, output_length, output_dir, output_name = r"\CNN_X_Test_Clean" + test_label)
        
        
        [pesq_score_Clean, pesq_score_Test, pesq_score_y_pred, snr_out]  = evaluate.eval2(Y_pred, X_test_phase, X_test_abs, X_test_clean_abs, X_test_clean_phase, alpha, threshold)
                                                        
        train_time = np.load(path + "train_time" + model_type + noise_type + "_" + str(snr) + "_dB.npy")
        test_time  = np.load(path +  "test_time" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

        print(str(snr) + " dB is" + str(snr_out))

        # % SAVE RESULTS IN MATRIX
        results_matrix_CNN[0,i] = snr
        results_matrix_CNN[1,i] = pesq_score_Clean
        results_matrix_CNN[2,i] = pesq_score_Test
        results_matrix_CNN[3,i] = pesq_score_y_pred
        results_matrix_CNN[4,i] = train_time
        results_matrix_CNN[5,i] = test_time
        results_matrix_CNN[6,i] = snr_out

        print("results of " + str(snr) + " dB done!" )

    print(results_matrix_CNN)

    output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Seed"
    #output_name = "\\results_matrix_CNN_Regression_"+ noise_type + "_" + str(segment_length/16000)
    #output_name = "\\results_array_CNN_Regression_"+ noise_type +  "_" + str(snr) + "_" + str(segment_length/16000)
    output_name = "\\results_matrix__CNN_Regression_"+ noise_type + "_" + str(segment_length/16000) + "_seed_" + str(seed)
    np.save(output_dir + output_name, results_matrix_CNN)

    # Export at matlab file
    functions.convert_npy_to_mat(output_dir)




    #  ESN Model 



    model_type = "ESN"

    path = r'C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\arrays_NPY_Drive\\'

    for i in range(len(snr_list)):

        snr = snr_list[i]

        X_train_abs          = np.load(path + "/X_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        Y_train_abs          = np.load(path + "/Y_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_abs           = np.load(path + "/X_Test_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_phase         = np.load(path + "/X_Test_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy")
        X_test_clean_abs     = np.load(path + "/X_Test_Clean_abs_split_" + noise_type + ".npy")
        X_test_clean_phase   = np.load(path + "/X_Test_Clean_phase_split_" + noise_type + ".npy")


        rpy.verbosity(0)
        rpy.set_seed(seed)

        # SLET IKKE DEN HER! SO-FAR THE GOLDEN STANDARD!
        #reservoir = Reservoir(1000, lr=0.5, sr=0.9)
        reservoir = Reservoir(1000, lr=0.5, sr=0.9)
        ridge = Ridge(ridge=1e-7)

        # Reshape to (2239, 256, 64)
        X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
        Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))

        print("NPY imported")
        print(X_train_abs.shape)
        print(Y_train_abs.shape)

        # Create the ESN model

        # Create a readout
        readout = Ridge(ridge=1e-7)

        #  Train the readout
        print("Training ESN" + str(snr))

        t=time.time()
        train_states = reservoir.run(X_train_abs[0], reset=True) # train-states are the activations of the reservoir triggered by the X_train
        readout = readout.fit(train_states, Y_train_abs[0], warmup=10) # Train the readout
        esn_model = reservoir >> ridge

        # Train the ESN
        esn_model = esn_model.fit(X_train_abs, Y_train_abs, warmup=10)
        train_time = time.time()-t

        print(snr)
        print(train_time)
        print(reservoir.is_initialized, readout.is_initialized, readout.fitted)

        # Test ESN

        X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
        X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))

        print("NPY imported")
        print(X_test_abs.shape)
        print(X_test_clean_abs.shape)

        t=time.time()
        Y_pred = esn_model.run(X_test_abs)
        test_time = time.time()-t

        print(test_time)

        Y_pred = np.array(Y_pred)
        print("Test run complete")
        print(Y_pred.shape)

        np.save(path + "Y_pred_" + model_type + noise_type + "_" + str(snr) + "_dB.npy", Y_pred)
        np.save(path + "train_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy", train_time)
        np.save(path + "test_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy", test_time)

        print(str(snr) +"_done")






    #  EVALUATE ESN MODEL OUTPUT




    results_matrix_ESN = np.zeros((7,len(snr_list)))
    path = r'C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\arrays_NPY_Drive\\'
    model_type = "ESN"


    # For SNR out calculation

    alpha = 0.01
    threshold = 0.1e8
    threshold = 0.02 # 09/01

    for i in range(len(snr_list)):

        snr = snr_list[i]

        # IMPORT NPY
        X_train_abs, Y_train_abs, X_test_abs, X_test_phase, X_test_clean_abs, X_test_clean_phase = functions.import_NPY_no_phase(path, snr, noise_type)

        Y_pred = np.load(path  + "Y_pred_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

        print("Y_pred max" + str(abs(np.max(Y_pred))))

        # Reshape 
        X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
        Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
        X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
        X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))
        Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], Y_pred.shape[1], Y_pred.shape[2]))

        # % POST PROCESS TO AUDIO
        output_length = segment_length/16000 
        input_length = segment_length
        output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\processed audio data"
        test_label = "_" + model_type + "_" + noise_type + "_" + str(snr) + "_" + r".wav"
        

        #Y_pred
        functions.concatenate_signal(Y_pred, X_test_phase, input_length, output_length, output_dir, output_name = r"\ESN_model_output" + test_label)
        # X_test
        functions.concatenate_signal(X_test_abs, X_test_phase, input_length, output_length, output_dir, output_name = r"\ESN_X_Test" + test_label)
        # X_test_clean
        functions.concatenate_signal(X_test_clean_abs, X_test_clean_phase, input_length, output_length, output_dir, output_name = r"\ESN_X_Test_Clean" + test_label)
        

        [pesq_score_Clean, pesq_score_Test, pesq_score_y_pred, snr_out]  = evaluate.eval2(Y_pred, X_test_phase, X_test_abs, X_test_clean_abs, X_test_clean_phase, alpha, threshold)
        print("Y_pred max" + str(abs(np.max(Y_pred))))                                                   
        train_time = np.load(path + "train_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")
        test_time  = np.load(path +  "test_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

        print(str(snr) + " dB is" + str(snr_out))

        # % SAVE RESULTS IN MATRIX
        results_matrix_ESN[0,i] = snr
        results_matrix_ESN[1,i] = pesq_score_Clean
        results_matrix_ESN[2,i] = pesq_score_Test
        results_matrix_ESN[3,i] = pesq_score_y_pred
        results_matrix_ESN[4,i] = train_time
        results_matrix_ESN[5,i] = test_time
        results_matrix_ESN[6,i] = snr_out

        print("results of " + str(snr) + " dB done!" )

    print(results_matrix_ESN)

    output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Seed"
    output_name = "\\results_matrix_ESN_Regression_"+ noise_type + "_" + str(segment_length/16000) + "_seed_" + str(seed)
    np.save(output_dir + output_name, results_matrix_ESN)

    # Export at matlab file
    functions.convert_npy_to_mat(output_dir)



print("SEED LOOP DONE!")

# LOOP END 





# %% TEST AF SNR PÅ INPUT

importlib.reload(functions)

snr_list = [10, 8, 6, 4, 2, 0, -2, -4, -6, -8, -10]

#snr_list = [-8]


results_matrix_ESN = np.zeros((7,len(snr_list)))
path = r'C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\arrays_NPY_Drive\\'
model_type = "ESN"


# For SNR out calculation
alpha = 0.01
alpha = 0.01
threshold = 0.1e8
threshold = 0.05 # 09/01
threshold = 0.02 # 09/01

for i in range(len(snr_list)):

    snr = snr_list[i]

    # IMPORT NPY
    X_train_abs, Y_train_abs, X_test_abs, X_test_phase, X_test_clean_abs, X_test_clean_phase = functions.import_NPY_no_phase(path, snr, noise_type)

    Y_pred = np.load(path  + "Y_pred_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

    print("Y_pred max" + str(abs(np.max(Y_pred))))

    # Reshape 
    X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
    Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
    X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
    X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))
    Y_pred = np.reshape(Y_pred, (Y_pred.shape[0], Y_pred.shape[1], Y_pred.shape[2]))

    # % POST PROCESS TO AUDIO
    output_length = segment_length/16000 
    input_length = segment_length
    output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\processed audio data"
    test_label = "_" + model_type + "_" + noise_type + "_" + str(snr) + "_" + r".wav"
    
    #Y_pred
    functions.concatenate_signal(Y_pred, X_test_phase, input_length, output_length, output_dir, output_name = r"\ESN_model_output" + test_label)
    # X_test
    functions.concatenate_signal(X_test_abs, X_test_phase, input_length, output_length, output_dir, output_name = r"\ESN_X_Test" + test_label)
    # X_test_clean
    functions.concatenate_signal(X_test_clean_abs, X_test_clean_phase, input_length, output_length, output_dir, output_name = r"\ESN_X_Test_Clean" + test_label)
    
    # 09/01
    #X_test_abs = X_test_abs[0:5]
    #X_test_clean_abs = X_test_clean_abs[0:5]
    #X_test_clean_phase = X_test_clean_phase[0:5]
    #X_test_clean_phase = X_test_clean_phase[0:5]


    [pesq_score_Clean, pesq_score_Test, pesq_score_y_pred, snr_out]  = evaluate.eval2(X_test_clean_abs, X_test_clean_abs, X_test_clean_abs, X_test_clean_abs, X_test_clean_phase, alpha, threshold)
    print("Y_pred max" + str(abs(np.max(Y_pred))))                                                   
    train_time = np.load(path + "train_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")
    test_time  = np.load(path +  "test_time_" + model_type + noise_type + "_" + str(snr) + "_dB.npy")

    print(str(snr) + " dB is" + str(snr_out))

    # % SAVE RESULTS IN MATRIX
    results_matrix_ESN[0,i] = snr
    results_matrix_ESN[1,i] = pesq_score_Clean
    results_matrix_ESN[2,i] = pesq_score_Test
    results_matrix_ESN[3,i] = pesq_score_y_pred
    results_matrix_ESN[4,i] = train_time
    results_matrix_ESN[5,i] = test_time
    results_matrix_ESN[6,i] = snr_out

    print("results of " + str(snr) + " dB done!" )

print(results_matrix_ESN)

output_dir = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results"
#output_name = "\\results_matrix_ESN_Regression_001_TEST"+ noise_type + "_" + str(segment_length/16000)
output_name = "\\results_SNR_of_Clean_signals_TEST"+ noise_type + "_" + str(segment_length/16000)
np.save(output_dir + output_name, results_matrix_ESN)

# Export at matlab file
functions.convert_npy_to_mat(output_dir)
#functions.convert_npy_to_mat(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\SNR plot data")



# %%


# CONVERT TO MATLAB

functions.convert_npy_to_mat(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Data fra ekstern\Speech SSL results")

# %% 

# F1 score 

import sklearn 

path = r"D:\Speciale\Speech SSL results\\"

score_all_cnn = np.load(path + "score_cnn_alldirections.npy")
score_16_cnn = np.load(path + "score_cnn_16directions.npy")
score_4_cnn = np.load(path + "score_cnn_4directions.npy")

Y_pred_all_cnn = np.load(path + "Y_pred_class_cnn_test_alldirections.npy")
Y_pred_16_cnn = np.load(path + "Y_pred_class_cnn_test_16directions.npy")
Y_pred_4_cnn = np.load(path + "Y_pred_class_cnn_test_4directions.npy")

Y_test_all_cnn = np.load(path + "Y_test_class_cnn_test_alldirections.npy")
Y_test_16_cnn = np.load(path + "Y_test_class_cnn_test_16directions.npy")
Y_test_4_cnn = np.load(path + "Y_test_class_cnn_test_4directions.npy")

f1_all_cnn = sklearn.metrics.f1_score(Y_test_all_cnn, Y_pred_all_cnn, labels=None, pos_label=1, average='weighted', sample_weight=None)
f1_16_cnn = sklearn.metrics.f1_score(Y_test_16_cnn, Y_pred_16_cnn, labels=None, pos_label=1, average='weighted', sample_weight=None)
f1_4_cnn = sklearn.metrics.f1_score(Y_test_4_cnn, Y_pred_4_cnn, labels=None, pos_label=1, average='weighted', sample_weight=None)



score_all_esn = np.load(path + "score_esn_alldirections.npy")
score_16_esn = np.load(path + "score_esn_16directions.npy")
score_4_esn = np.load(path + "score_esn_4directions.npy")

Y_pred_all_esn = np.load(path + "Y_pred_class_esn_test_alldirections.npy")
Y_pred_16_esn = np.load(path + "Y_pred_class_esn_test_16directions.npy")
Y_pred_4_esn = np.load(path + "Y_pred_class_esn_test_4directions.npy")

Y_test_all_esn = np.load(path + "Y_test_class_esn_test_alldirections.npy")
Y_test_16_esn = np.load(path + "Y_test_class_esn_test_16directions.npy")
Y_test_4_esn = np.load(path + "Y_test_class_esn_test_4directions.npy")

f1_all_esn = sklearn.metrics.f1_score(Y_test_all_esn, Y_pred_all_esn, labels=None, pos_label=1, average='weighted', sample_weight=None)
f1_16_esn = sklearn.metrics.f1_score(Y_test_16_esn, Y_pred_16_esn, labels=None, pos_label=1, average='weighted', sample_weight=None)
f1_4_esn = sklearn.metrics.f1_score(Y_test_4_esn, Y_pred_4_esn, labels=None, pos_label=1, average='weighted', sample_weight=None)

print("cnn")
print(score_all_cnn)
print(score_16_cnn)
print(score_4_cnn)
print(f1_all_cnn)
print(f1_16_cnn)
print(f1_4_cnn)

print("esn:")
print(score_all_esn)
print(score_16_esn)
print(score_4_esn)
print(f1_all_esn)
print(f1_16_esn)
print(f1_4_esn)
# %%
