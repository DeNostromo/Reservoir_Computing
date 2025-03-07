#  %% Functions 

import os
import numpy as np
import soundfile as sf
from scipy.io import wavfile
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal
from scipy.io import wavfile
from scipy.signal import stft, istft
import scipy as sc
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from scipy.io.wavfile import write
import soundfile as sf
import os
import random as rnd
import shutil
import matlab.engine
from natsort import natsorted
import colorednoise as cn
from sympy import symbols, Eq, solve, I
import scipy.io as sio

# Change the current working directory to the script's directory
os.chdir(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment")


def add_noise(signal, noise_factor, noise_type):
    """
    Adds white noise to the given signal.
    
    :param signal: The original audio signal (numpy array).
    :param noise_factor: The factor controlling the amount of noise (float).
    :return: The noisy signal (numpy array).
    """
    

    if noise_type == "white_noise":
        noise = np.random.normal(0, noise_factor, signal.shape)
        noisy_signal = signal + noise
        
        P_noise = 1/len(noise)*sum(noise**2)

        #print("P_noise is: ", P_noise)

    elif noise_type == "pink_noise":
        beta = 1 # the exponent
        samples = len(signal) # number of samples to generate
        noise = cn.powerlaw_psd_gaussian(beta, samples)
        noisy_signal = signal + noise

    return noisy_signal, P_noise

def add_noise_cont_SNR(signal, noise_type, snr):

    SNR = db_to_mag(snr)
    n = len(signal)
    P_y = 1/n*sum(signal**2)

    if noise_type == "white_noise":

        noise = np.random.normal(0, 1, n)
        noise_factor = np.sqrt(P_y*n/(SNR*sum(noise**2)))

        noise = noise*noise_factor

        noisy_signal = signal + noise
        #print("amp before noise sig:" + str(max(abs(signal))))

        #print("amp after noise sig:" + str(max(abs(noisy_signal))))
        
        P_noise = 1/len(noise)*sum(noise**2)

        #snr_of_sig = 10*np.log10(P_y / P_noise)
        #print("snr is:", snr_of_sig)


    elif noise_type == "pink_noise":
        beta = 1 # the exponent
        samples = len(signal) # number of samples to generate
        noise = cn.powerlaw_psd_gaussian(beta, samples)

        noise_factor = np.sqrt(P_y*n/(SNR*sum(noise**2)))
        noise = noise*noise_factor

        noisy_signal = signal + noise

        P_noise = 1/len(noise)*sum(noise**2)

    return noisy_signal, P_noise

def add_noise_from_file(signal, noise_factor, noise_type):

    if noise_type == "white_noise":
        path_noise = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\Noise\whiteNoise_audacity.wav"

    elif noise_type == "pink_noise":
        path_noise = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\Noise\pinkNoise_audacity.wav"

    noise, sample_rate_noise =sf.read(path_noise)
    noise = normalize_audio(noise)

    rand_index = rnd.randint(0,len(noise)-len(signal)-1)
    noise = noise[rand_index:rand_index+len(signal)]

    noisy_signal = signal + noise*noise_factor

    # rfft to confirm noise
    # spec_white_noise = sc.fft.rfft(noise)

    # plt.figure()
    # plt.plot(10*np.log10(abs(spec_white_noise)))

    return noisy_signal

def add_pink_noise_from_file(signal, noise_factor, segment_length):

    path_noise = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\Noise\pinkNoise_audacity.wav"
    noise, sample_rate_noise =sf.read(path_noise)
    noise = normalize_audio(noise)

    rand_index = rnd.randint(0,len(noise)-len(signal)-1)
    noise = noise[rand_index:rand_index+len(signal)]

    print(len(signal))
    print(len(noise))

    noisy_signal = signal + noise*noise_factor

    # rfft to confirm noise
    # spec_white_noise = sc.fft.rfft(noise)

    # plt.figure()
    # plt.plot(10*np.log10(abs(spec_white_noise)))

    return noisy_signal

def normalize_audio(signal, max_amplitude=1.0):
    """
    Normalizes the audio signal to prevent clipping.
    
    :param signal: The audio signal (numpy array).
    :param max_amplitude: The maximum amplitude for normalization (default is 1.0 for floating-point audio).
    :return: The normalized signal (numpy array).
    """
    norm_sig = signal / np.max(np.abs(signal)) * max_amplitude
    # Normalize realtive to energy in signal
    

    return norm_sig

def pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length):

    CNN_data = "true"

    # Define STFT length
    nperseg = 512
    # Define overlap, 50 %
    noverlap = nperseg/2
    # fs 
    sample_rate = 16e3
    # the length of each sample in seconds
    #segment_length = 0.5

    total_data_matrix_abs = []
    total_data_matrix_phase = []

    # List of files in the input dir
    train_folder = natsorted(os.listdir(INPUT_DIR))
    #print(train_folder)

    for i in range(0, len(sorted(os.listdir(INPUT_DIR)))):
        # read audio
        sig, sample_rate = sf.read(INPUT_DIR +'\\' + train_folder[i])

        #if len(sig) < segment_length*sample_rate:
        if len(sig) < segment_length:
            print("Signal to short", i)

        else: 
            #print(i)
            #print("Signal long enough")
            
            # Trim signal to min time 
            #sig = sig[0:min_time*sample_rate]

            # stft 
            f_stft, t_stft, Zxx = stft(sig, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming")

            Zxx = np.delete(Zxx, 0, axis=0)

            if CNN_data == "true":
                Zxx = np.delete(Zxx, -1, axis=1)
            
            # abs and phase
            Zxx_abs = abs(Zxx)
            Zxx_phase = np.angle(Zxx)
            
            # Convert to logarithmic 
            Zxx_log = 10 * np.log10(Zxx_abs)
            
            # Scale
            scale_factor = 90
            Zxx_log = Zxx_log/scale_factor
            Zxx_log = Zxx_log +1

            
            #print(Zxx_log.shape)
            

            # Give it the extra dimension
            n_timeBin = Zxx_log.shape[1]
            Zxx_log = np.reshape(Zxx_log, (256, n_timeBin, 1))

            # Save in total data matrix
            total_data_matrix_abs.append(Zxx_log)
            total_data_matrix_phase.append(Zxx_phase)
            

    np.save(OUTPUT_DIR + output_name_abs, total_data_matrix_abs)
    np.save(OUTPUT_DIR + output_name_phase, total_data_matrix_phase)

    return total_data_matrix_abs, total_data_matrix_phase,  t_stft, f_stft, scale_factor

def pre_process_data_no_phase(INPUT_DIR, OUTPUT_DIR, output_name_abs, segment_length):

    CNN_data = "true"

    # Define STFT length
    nperseg = 512
    # Define overlap, 50 %
    noverlap = nperseg/2
    # fs 
    sample_rate = 16e3
    # the length of each sample in seconds
    #segment_length = 0.5

    total_data_matrix_abs = []

    # List of files in the input dir
    train_folder = natsorted(os.listdir(INPUT_DIR))
    #print(train_folder)

    for i in range(0, len(sorted(os.listdir(INPUT_DIR)))):
        # read audio
        sig, sample_rate = sf.read(INPUT_DIR +'\\' + train_folder[i])
        #print("amp sig before stft:" + str(max(abs(sig))))

        #if len(sig) < segment_length*sample_rate:
        if len(sig) < segment_length:
            print("Signal to short", i)
            print("short sig:" + str(max(abs(sig))))
        else: 
            #print(i)
            #print("Signal long enough")
            
            # Trim signal to min time 
            #sig = sig[0:min_time*sample_rate]

            # stft 
            f_stft, t_stft, Zxx = stft(sig, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming")

            Zxx = np.delete(Zxx, 0, axis=0)

            if CNN_data == "true":
                Zxx = np.delete(Zxx, -1, axis=1)
            
            # abs and phase
            Zxx_abs = abs(Zxx)
            Zxx_phase = np.angle(Zxx)
            
            # Convert to logarithmic 
            print(abs(np.max(Zxx_abs)))
            #print("amp sig after abs:" + str(np.max(Zxx_abs)))
            Zxx_log = 10 * np.log10(Zxx_abs)
            #print("amp sig after log:" + str(np.max(abs(Zxx_log))))
            # Scale
            scale_factor = 90
            Zxx_log = Zxx_log/scale_factor
            Zxx_log = Zxx_log +1
            #print("amp sig after scale:" + str(np.max(abs(Zxx_log))))

            
            #print(Zxx_log.shape)
            

            # Give it the extra dimension
            n_timeBin = Zxx_log.shape[1]
            Zxx_log = np.reshape(Zxx_log, (256, n_timeBin, 1))

            # Save in total data matrix
            total_data_matrix_abs.append(Zxx_log)

            

    np.save(OUTPUT_DIR + output_name_abs, total_data_matrix_abs)


    return total_data_matrix_abs, t_stft, f_stft, scale_factor

def split_audio_into_segments(input_folder, output_folder, segment_length):

    # Get a list of all audio files in the input folder
    audio_files = [f for f in os.listdir(input_folder) if f.endswith(('.wav', '.mp3', '.flac', '.ogg'))]

    for audio_file in audio_files:
        # Load the audio file
        file_path = os.path.join(input_folder, audio_file)
        y, sr = sf.read(file_path)

        # Calculate the number of samples per segment
        #samples_per_segment = int(segment_length * sr)
        samples_per_segment = int(segment_length)


        # if len(y) < samples_per_segment: 
        #     print("signal too short split")


        # Split the audio into segments
        num_segments = len(y) // samples_per_segment
        for i in range(num_segments):
            start = i * samples_per_segment
            end = start + samples_per_segment
            segment = y[start:end]

            # Save the segment as a new audio file
            segment_filename = f"{os.path.splitext(audio_file)[0]}_segment_{i}.wav"
            segment_filepath = os.path.join(output_folder, segment_filename)
            sf.write(segment_filepath, segment, sr)

            #print(f"Saved segment {i} of {audio_file} as {segment_filename}")

def delete_folder_contents(folder_path):

    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def normalize_folder(input_folder, OUTPUT_DIR):

    for filename in os.listdir(input_folder):
        input_file = os.path.join(input_folder, filename)
        sig, sample_rate = sf.read(input_file)
        #print("amp sig before norm:" + str(max(abs(sig))))
        sig = normalize_audio(sig)
        #print("amp sig after norm:" + str(max(abs(sig))))
        sf.write(OUTPUT_DIR + "\\" + str(filename), sig, sample_rate)    
    return

def process_flac_files(input_folder, output_folder, noise_factor, noise_type, snr):
    """
    Processes all FLAC files in the input folder, adds white noise, and saves the results in the output folder.
    
    :param input_folder: Path to the input folder containing FLAC files.
    :param output_folder: Path to save the output FLAC files.
    :param noise_factor: The factor controlling the amount of noise (float).
    """
    # Create the output folder if it doesn't exist
    # if not os.path.exists(output_folder):
    #     os.makedirs(output_folder)
    
    i = 0
    P_noise_avg = 0

    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".flac"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, filename)
            i += 1
            # Read the input FLAC file
            signal, sample_rate = sf.read(input_file)

            # Normalize the noisy signal to prevent clipping
            #signal = normalize_audio(signal)
            
            # Add white noise to the signal
            #noisy_signal = add_pink_noise_from_file(signal, noise_factor)

            noisy_signal, P_noise = add_noise_cont_SNR(signal, noise_type, snr)
            P_noise_avg = (P_noise_avg + P_noise)
            #print("avg", P_noise_avg)

            # Normalize the noisy signal to prevent clipping
            #normalized_signal = normalize_audio(noisy_signal)
            
            # Save the noisy signal to a new FLAC file
            sf.write(output_file, noisy_signal, sample_rate)
            #print(f"Processed and saved: {output_file}")

    P_noise_avg = P_noise_avg/i
  


    return P_noise_avg

def convert_flac_to_wav(input_folder, output_folder):
    """
    Converts all FLAC files in the input folder to WAV files and saves them in the output folder.
    
    :param input_folder: Path to the input folder containing FLAC files.
    :param output_folder: Path to save the output WAV files.
    """
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".flac"):
            input_file = os.path.join(input_folder, filename)
            output_file = os.path.join(output_folder, os.path.splitext(filename)[0] + ".wav")
            
            # Read the FLAC file
            data, sample_rate = sf.read(input_file)
            #print("amp sig before wav:" + str(max(abs(data))))

            # Convert data to 16-bit integer format 
            data = (data * 32767).astype(np.int16)
            
            # Write the data to a WAV file
            wavfile.write(output_file, sample_rate, data)
            #print(f"Converted and saved: {output_file}")

def post_process_data(input_abs, input_phase):

    print("HEY")

    #input_abs = Y_test[0][:,:,0]
    #input_phase = X_test_phase[0]

    nperseg = 512
    noverlap = nperseg/2
    sample_rate = int(16e3)

    # scale back 
    input_abs = input_abs * 90

    # Convert back to linear 
    convert_back_matrix_abs_lin = 10 ** (input_abs / 10)

    print("abs matrix:", convert_back_matrix_abs_lin.shape)
    print("phase matrix:", input_phase.shape)

    # Combine magnitude and phase to form the complex spectrogram
    complex_spectrogram = convert_back_matrix_abs_lin * np.exp(1j * input_phase)
    print(np.array(complex_spectrogram).shape)
    # ISTFT
    t , convert_to_time = istft(complex_spectrogram, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window="hamming")

    # Trim signal
    convert_to_time = convert_to_time[0:16000]
    


    #convert_to_time = convert_to_time/max(convert_to_time)

    #sf.write(output_dir + output_name, convert_to_time, sample_rate, 'PCM_16')

    return convert_to_time

def post_process_data_array_split(input_abs, input_phase):

    #input_abs = Y_test[0][:,:,0]
    #input_phase = X_test_phase[0]

    nperseg = 512
    noverlap = nperseg/2
    sample_rate = int(16e3)

    # scale back 
    #input_abs = input_abs -1 # 09/01
    input_abs = input_abs * 90

    # Convert back to linear 
    convert_back_matrix_abs_lin = 10 ** (input_abs / 10)

    # Combine magnitude and phase to form the complex spectrogram
    complex_spectrogram = convert_back_matrix_abs_lin * np.exp(1j * input_phase)

    # Add empty top freq vector 
    n_collums = complex_spectrogram.shape[1]

    top_freq_row = np.zeros(n_collums)
    complex_spectrogram = np.vstack((top_freq_row, complex_spectrogram))

    # ISTFT
    t , convert_to_time = istft(complex_spectrogram, fs=sample_rate, nperseg=nperseg, noverlap=noverlap, window = "hamming")

    # Trim signal
    #convert_to_time = convert_to_time[0:16000]

    return convert_to_time

def concatenate_signal(matrix_abs, matrix_phase, input_length, output_length, output_dir, output_name):

    #input_length = 1
    con_result_abs = np.empty(0)
    sampling_frequency = 16000
    s = output_length//(input_length/sampling_frequency)
    input_samples = input_length

    for i in range(int(s)):

        convert_back_matrix_abs = matrix_abs[i][:,:]
        convert_back_matrix_phase = matrix_phase[i]

        time_sig = post_process_data_array_split(convert_back_matrix_abs, convert_back_matrix_phase)
        time_sig = time_sig[:input_samples]
        #print("time_sig is:" + str(time_sig))
        con_result_abs = np.hstack((con_result_abs, time_sig))

    #time_sig = post_process_data_array_split(convert_back_matrix_abs, convert_back_matrix_phase, output_dir, output_name)
    con_result_abs = con_result_abs/max(abs(con_result_abs))
    sf.write(output_dir + output_name, con_result_abs, 16000, 'PCM_16')
    print("Signal exported")

    #return con_result_abs

def concatenate_signal_folder(matrix_abs, matrix_phase, input_length, output_length, output_dir, output_name):

    #input_length = 1
    con_result_abs = np.empty(0)
    sampling_frequency = 16000
    s = output_length//(input_length/sampling_frequency)
    input_samples = input_length

    for i in range(int(s)):

        convert_back_matrix_abs = matrix_abs[i][:,:]
        convert_back_matrix_phase = matrix_phase[i]

        time_sig = post_process_data_array_split(convert_back_matrix_abs, convert_back_matrix_phase)
        time_sig = time_sig[:input_samples]
        con_result_abs = np.hstack((con_result_abs, time_sig))

    
    #time_sig = post_process_data_array_split(convert_back_matrix_abs, convert_back_matrix_phase, output_dir, output_name)
    con_result_abs = con_result_abs/max(abs(con_result_abs))
    sf.write(output_dir + output_name, con_result_abs, 16000, 'PCM_16')
    print("Signal exported")

def import_NPY(path, snr, noise_type):
     # Import pre-processed data from NPY 
    #NPY_path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\CNN REGRESSION MODEL\data\arrays_NPY\\"
    NPY_path = path

    # X_Train 
    abs_input = NPY_path + r"X_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    abs_phase =  NPY_path + r"X_Train_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    X_train_abs = np.load(abs_input)
    X_train_phase = np.load(abs_phase)
    # Y_Train
    abs_input = NPY_path + r"Y_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    abs_phase = NPY_path + r"Y_Train_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    Y_train_abs = np.load(abs_input)
    Y_train_phase = np.load(abs_phase)
    # X_Test
    abs_input = NPY_path + r"X_Test_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    abs_phase = NPY_path + r"X_Test_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    X_test_abs = np.load(abs_input)
    X_test_phase = np.load(abs_phase)
    # X_Test_clean
    abs_input = NPY_path + r"X_Test_Clean_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    abs_phase = NPY_path + r"X_Test_Clean_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    X_test_clean_abs = np.load(abs_input)
    X_test_clean_phase = np.load(abs_phase)

    # Reshape to (2239, 256, 64)
    X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
    Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
    X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
    X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))

    print("NPY imported")
    return X_train_abs, X_train_phase, Y_train_abs, Y_train_phase, X_test_abs, X_test_phase, X_test_clean_abs, X_test_clean_phase

def import_NPY_no_phase(path, snr, noise_type):
     # Import pre-processed data from NPY 
    #NPY_path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\CNN REGRESSION MODEL\data\arrays_NPY\\"
    NPY_path = path

    # X_Train 
    abs_input = NPY_path + r"X_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    #abs_phase =  NPY_path + r"X_Train_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    X_train_abs = np.load(abs_input)
    #X_train_phase = np.load(abs_phase)
    # Y_Train
    abs_input = NPY_path + r"Y_Train_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    #abs_phase = NPY_path + r"Y_Train_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    Y_train_abs = np.load(abs_input)
    #Y_train_phase = np.load(abs_phase)
    # X_Test
    abs_input = NPY_path + r"X_Test_abs_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    abs_phase = NPY_path + r"X_Test_phase_split_" + noise_type + "_" + str(snr)+ "_dB" + ".npy"
    X_test_abs = np.load(abs_input)
    X_test_phase = np.load(abs_phase)
    # X_Test_clean
    abs_input = NPY_path + r"X_Test_Clean_abs_split_" + noise_type + ".npy"
    abs_phase = NPY_path + r"X_Test_Clean_phase_split_" + noise_type + ".npy"
    X_test_clean_abs = np.load(abs_input)
    X_test_clean_phase = np.load(abs_phase)

    # Reshape 
    X_train_abs = np.reshape(X_train_abs, (X_train_abs.shape[0], X_train_abs.shape[1], X_train_abs.shape[2]))
    Y_train_abs = np.reshape(Y_train_abs, (Y_train_abs.shape[0], Y_train_abs.shape[1], Y_train_abs.shape[2]))
    X_test_abs = np.reshape(X_test_abs, (X_test_abs.shape[0], X_test_abs.shape[1], X_test_abs.shape[2]))
    X_test_clean_abs = np.reshape(X_test_clean_abs, (X_test_clean_abs.shape[0], X_test_clean_abs.shape[1], X_test_clean_abs.shape[2]))

    print("NPY imported")
    return X_train_abs, Y_train_abs, X_test_abs, X_test_phase, X_test_clean_abs, X_test_clean_phase

def SNR(y, noise, noise_factor):
    
    sig_length = min((len(y), len(noise)))
    y = y[0:sig_length]
    #noise = noise[0:sig_length] * noise_factor
    noise = noise[0:len(y)]*noise_factor
    P_clean = 1/len(y)*sum(y**2)
    P_noise = 1/len(y)*sum(noise**2)

    SNR = 10*np.log10(P_clean/P_noise)

    return SNR

def SNR_of_folder(folder_path, noise_type, noise_factor):

    if noise_type == "white_noise":
        path_noise = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\Noise\whiteNoise_audacity.wav"
    elif noise_type == "pink_noise":
        path_noise = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\Noise\pinkNoise_audacity.wav"

    snr_values = []

     # Iterate over all files in the folder
    for filename in os.listdir(folder_path):

        if filename.endswith(".wav"):
            file_path = os.path.join(folder_path, filename)
            
            # Read the audio file
            y, sample_rate = sf.read(file_path)

            # Read the noise file
            noise, sample_rate_noise =sf.read(path_noise)
            noise = normalize_audio(noise)
            rand_noise = np.empty(len(y))

            for i in range(len(y)):
                rand_index = rnd.randint(0,len(y)-1)
                rand_noise[i]=noise[rand_index]

            # Bypass
            #rand_noise = noise

            # Calculate the SNR for the audio file
            snr = SNR(y, rand_noise, noise_factor)
            
            # Append the SNR value to the list
            snr_values.append(snr)

    # Calculate the average SNR
    average_snr = np.mean(snr_values)
    
    return average_snr

def db_to_mag(db_value):
    return 10 ** (db_value / 10)

def mag_to_db(mag_value):
    return 10 * np.log10(mag_value)

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

def SNR_of_output(Y_pred_time, Clean_time, X_test_abs,  threshold, alpha, plot, flag):
    
    #print(len(Y_pred_time))
    #print(len(Clean_time))


    clean_abs = abs(Clean_time)

    res = np.zeros(len(clean_abs))
    mask = np.ones(len(res))
    y_smooth = 0
    

    for i in range(len(clean_abs)):

        y_smooth = alpha * clean_abs[i] + (1-alpha)*y_smooth
        res[i] = y_smooth


    indices_noise = np.argwhere(res < threshold)
    pure_noise = Y_pred_time[indices_noise]
    #print("pure_noise" + str(len(pure_noise)))
    #print(indices_noise)

    P_pure_noise = 1/len(pure_noise) *sum(pure_noise**2)

    P_sig = 1/len(Y_pred_time) * sum(Y_pred_time**2)

    P_speech = P_sig - P_pure_noise

    path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\SNR plot data"
    # np.save(path + r"\Clean_SNR_data", Clean_time)
    # np.save(path + r"\ref_SNR_data", res)
    # np.save(path + r"\Y_pred_time_SNR_data", Y_pred_time)
    # np.save(path + r"\SNR_indices_good", indices_noise)

    if plot == 1:
  
        plt.figure()
        plt.plot(Clean_time)
        plt.axhline(y = threshold, color = 'r', linestyle = '-') 
    
        plt.figure()
        plt.plot(res)
        plt.axhline(y = threshold, color = 'r', linestyle = '-') 

        plt.figure()
        plt.plot(Y_pred_time)
        plt.plot(indices_noise, Y_pred_time[indices_noise], color="r")
        plt.axhline(y = threshold, color = 'y', linestyle = '-') 


    

    if (P_speech <= 0): 
        print("P_sig: " + str(P_sig))
        print("P_speech: " + str(P_speech))
        print("P_noise: " + str(P_pure_noise))

        # plt.figure()
        # plt.plot(res)
        # plt.axhline(y = threshold, color = 'r', linestyle = '-') 
        #print(Clean_time.shape)

        if (flag ==0):
            plt.figure()
            plt.plot(X_test_abs)
            plt.axhline(y = threshold, color = 'g', linestyle = '-')

            plt.figure()
            plt.plot(Clean_time)
            plt.axhline(y = threshold, color = 'r', linestyle = '-')

            plt.figure()
            plt.plot(res)
            plt.axhline(y = threshold, color = 'b', linestyle = '-')

            plt.figure()
            plt.plot(Y_pred_time)
            plt.plot(indices_noise, Y_pred_time[indices_noise], color="r")
            plt.axhline(y = threshold, color = 'y', linestyle = '-') 

            np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Negative SNR results\\" + "clean_time", Clean_time)
            np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Negative SNR results\\" + "res", res)
            np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Negative SNR results\\" + "Y_pred_time", Y_pred_time)
            np.save(r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\results\Negative SNR results\\" + "Indices", indices_noise)






        #Y_pred_time = Y_pred_time / max(abs(Y_pred_time))

        output = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\processed audio data\SNR tests\test_audio.wav" 
        sf.write(output, Y_pred_time, 16000, 'PCM_16')
        print("Signal exported")

        flag = 1
        n_ignore = 1
        snr = 0

    elif (P_speech > 0):
        snr = 10*np.log10(P_speech/P_pure_noise)
        #print("snr test " + str(snr))
        #snr = 10*np.log10(P_sig/P_pure_noise)
        n_ignore = 0

    #snr = 10*np.log10(P_sig/P_pure_noise)
    #snr = 10*np.log10(P_sig) - 10*np.log10(P_pure_noise)

    # Quick and dirty kode! Hvad gør man hvis P_speech er negativ, altså en negativ SNR? Det kan log ikke klare? 

    # if P_speech <= 0:
    #     snr = 0

    # else:
    #     snr = 10*np.log10(P_speech / P_pure_noise)
   

    return snr, flag, n_ignore


# %%
