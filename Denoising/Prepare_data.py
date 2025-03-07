# %% Prepare data
import functions
import functions_NPY
import numpy as np
import importlib

importlib.reload(functions)
importlib.reload(functions_NPY)

def make_data(path, noise_factor, segment_length, noise_type, snr):

    # Noise factor
    #noise_factor = 0.01

    # Segment length in seconds
    #segment_length = 1 

    #path = r"C:\Users\sed\OneDrive - Invisio\Documents\Sixten Speciale\Regression environment\data\\"

    # Train data
    #  - LONG
    # Normalize original
    input_folder = path + r"Train - Clean speech audio flac original - LONG"
    output_folder = path + r"Train - Clean speech audio flac normalized"
    functions.delete_folder_contents(output_folder)
    functions.normalize_folder(input_folder, output_folder) 

    # Noise 
    input_folder =  path + r"Train - Clean speech audio flac normalized" 
    output_folder = path + r"Train - Noisy speech audio flac"  
    functions.delete_folder_contents(output_folder)
    functions.process_flac_files(input_folder, output_folder, noise_factor, noise_type, snr)

    # folder to wav - clean 
    input_folder =  path + r"Train - Clean speech audio flac normalized" 
    output_folder = path + r"Train - Clean speech audio wav"  
    functions.delete_folder_contents(output_folder)
    functions.convert_flac_to_wav(input_folder, output_folder)

    # Folder to wav Train - noisy
    input_folder =  path + r"Train - Noisy speech audio flac" 
    output_folder = path + r"Train - Noisy speech audio wav"  
    functions.delete_folder_contents(output_folder)
    functions.convert_flac_to_wav(input_folder, output_folder)

    # Split data clean
    input_folder = path + r"Train - Clean speech audio wav"
    output_folder = path + r"Train - Clean speech audio wav split"
    functions.delete_folder_contents(output_folder)
    functions.split_audio_into_segments(input_folder, output_folder, segment_length)

    # Split data noisy
    input_folder = path + r"Train - Noisy speech audio wav"
    output_folder = path + r"Train - Noisy speech audio wav split"
    functions.delete_folder_contents(output_folder)
    functions.split_audio_into_segments(input_folder, output_folder, segment_length)


    # TEST DATA

    # Normalize clean speech wav
    input_folder =  path + r"Test - Clean speech audio flac original - LONG"  # Replace with your input folder path
    output_folder = path + r"Test - Clean speech audio flac normalized"  # Replace with your desired output folder path
    functions.delete_folder_contents(output_folder)
    functions.normalize_folder(input_folder, output_folder) 

    # Noise
    input_folder =  path + r"Test - Clean speech audio flac normalized"  # Replace with your input folder path
    output_folder = path + r"Test - Noisy speech audio flac"  # Replace with your desired output folder path
    functions.delete_folder_contents(output_folder)
    P_noise_avg = functions.process_flac_files(input_folder, output_folder, noise_factor, noise_type, snr)

    # Folder to wav clean
    input_folder =  path + r"Test - Clean speech audio flac normalized"  # Replace with your input folder path
    output_folder = path + r"Test - Clean speech audio wav"  # Replace with your desired output folder path
    functions.delete_folder_contents(output_folder)
    functions.convert_flac_to_wav(input_folder, output_folder)

    # Folder to wav noisy
    input_folder =  path + r"Test - Noisy speech audio flac"  # Replace with your input folder path
    output_folder = path + r"Test - Noisy speech audio wav"  # Replace with your desired output folder path
    functions.delete_folder_contents(output_folder)
    functions.convert_flac_to_wav(input_folder, output_folder)

    # Split data clean
    input_folder = path + r"Test - Clean speech audio wav"
    output_folder = path + r"Test - Clean speech audio wav split"
    functions.delete_folder_contents(output_folder)
    functions.split_audio_into_segments(input_folder, output_folder, segment_length)

    # Split data noisy
    input_folder = path + r"Test - Noisy speech audio wav"
    output_folder = path + r"Test - Noisy speech audio wav split"
    functions.delete_folder_contents(output_folder)
    functions.split_audio_into_segments(input_folder, output_folder, segment_length)

    # print("")
    # print("Done making noise")


    # # Save split signals as NPY

    # #  X_TRAIN SPLIT
    # #Set paths to input and output data for X_TRAIN
    # INPUT_DIR  = path + r"Train - Noisy speech audio wav split"
    # OUTPUT_DIR = path + r"arrays_NPY"
    # output_name_abs = "\\X_Train_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # output_name_phase = "\\X_Train_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # X_Train_abs_split, X_Train_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)


    # #  Y_TRAIN SPLIT
    # #Set paths to input and output data for Y_TRAIN
    # INPUT_DIR  = path + r"Train - Clean speech audio wav split"
    # OUTPUT_DIR = path + r"arrays_NPY"
    # output_name_abs = "\\Y_Train_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # output_name_phase = "\\Y_Train_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # Y_Train_abs_split, Y_Train_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)

    
    # #  X_TEST SPLIT
    # #Set paths to input and output data for X_TEST
    # INPUT_DIR  = path + r"Test - Noisy speech audio wav split"
    # OUTPUT_DIR = path + r"arrays_NPY"
    # output_name_abs = "\\X_Test_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # output_name_phase = "\\X_Test_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # X_Test_abs_split, X_Test_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)
 


    # #  X_TEST_CLEAN SPLIT
    # #Set paths to input and output data for X_TEST
    # INPUT_DIR  = path + r"Test - Clean speech audio wav split"
    # OUTPUT_DIR = path + r"arrays_NPY"
    # output_name_abs = "\\X_Test_Clean_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # output_name_phase = "\\X_Test_Clean_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # X_Test_Clean_abs_split, X_Test_Clean_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)


    # print("Converted to NPY")


    # Repeat for Drive ! 

    # Delete NPY drive folder content


        #  X_TRAIN SPLIT
    #Set paths to input and output data for X_TRAIN
    INPUT_DIR  = path + r"Train - Noisy speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY_Drive"
    output_name_abs = "\\X_Train_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
   # output_name_phase = "\\X_Train_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    X_Train_abs_split, t_stft, f_stft, scale_factor = functions.pre_process_data_no_phase(INPUT_DIR, OUTPUT_DIR, output_name_abs, segment_length)


    #  Y_TRAIN SPLIT
    #Set paths to input and output data for Y_TRAIN
    INPUT_DIR  = path + r"Train - Clean speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY_Drive"
    output_name_abs = "\\Y_Train_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    #output_name_phase = "\\Y_Train_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    Y_Train_abs_split, t_stft, f_stft, scale_factor = functions.pre_process_data_no_phase(INPUT_DIR, OUTPUT_DIR, output_name_abs, segment_length)

    
    #  X_TEST SPLIT
    #Set paths to input and output data for X_TEST
    INPUT_DIR  = path + r"Test - Noisy speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY_Drive"
    output_name_abs = "\\X_Test_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    output_name_phase = "\\X_Test_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    X_Test_abs_split, X_Test_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)
 

    # X_TEST_CLEAN MOVED OUTSIDE LOOP AS IT IS ONLY NEEDED ONCE 
    # #  X_TEST_CLEAN SPLIT
    # #Set paths to input and output data for X_TEST
    # INPUT_DIR  = path + r"Test - Clean speech audio wav split"
    # OUTPUT_DIR = path + r"arrays_NPY_Drive"
    # output_name_abs = "\\X_Test_Clean_abs_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # output_name_phase = "\\X_Test_Clean_phase_split_"+ noise_type + "_" + str(snr)+ "_dB"
    # X_Test_Clean_abs_split, X_Test_Clean_phase_split, t_stft, f_stft, scale_factor = functions.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)


    # print("Converted to NPY Drive")


    # Print shapes

    X_Train_abs_split = np.array(X_Train_abs_split)
    Y_Train_abs_split = np.array(Y_Train_abs_split)
    X_Test_abs_split = np.array(X_Test_abs_split)
    #X_Test_Clean_abs_split = np.array(X_Test_Clean_abs_split)
    print(X_Train_abs_split.shape)
    print(Y_Train_abs_split.shape)
    print(X_Test_abs_split.shape)
    #print(X_Test_Clean_abs_split.shape)

    print("Data done for " + str(snr))

    return P_noise_avg

def make_data_NPY(path, noise_factor, segment_length, noise_type, snr):

    # Train data
    #  - LONG
    # Normalize original
    input_folder = path + r"Train - Clean speech audio flac original - LONG"
    #output_folder = path + r"Train - Clean speech audio flac normalized"
    #functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.normalize_folder(input_folder, output_folder) 

    # Noise 
    input_folder =  path + r"Train - Clean speech audio flac normalized" 
    output_folder = path + r"Train - Noisy speech audio flac"  
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.process_flac_files(input_folder, output_folder, noise_factor, noise_type, snr)

    # folder to wav - clean 
    input_folder =  path + r"Train - Clean speech audio flac normalized" 
    output_folder = path + r"Train - Clean speech audio wav"  
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.convert_flac_to_wav(input_folder, output_folder)

    # Folder to wav Train - noisy
    input_folder =  path + r"Train - Noisy speech audio flac" 
    output_folder = path + r"Train - Noisy speech audio wav"  
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.convert_flac_to_wav(input_folder, output_folder)

    # Split data clean
    input_folder = path + r"Train - Clean speech audio wav"
    output_folder = path + r"Train - Clean speech audio wav split"
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.split_audio_into_segments(input_folder, output_folder, segment_length)

    # Split data noisy
    input_folder = path + r"Train - Noisy speech audio wav"
    output_folder = path + r"Train - Noisy speech audio wav split"
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.split_audio_into_segments(input_folder, output_folder, segment_length)



    # TEST DATA

    # Normalize clean speech wav
    input_folder =  path + r"Test - Clean speech audio flac original"  # Replace with your input folder path
    output_folder = path + r"Test - Clean speech audio flac normalized"  # Replace with your desired output folder path
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.normalize_folder(input_folder, output_folder) 

    # Noise
    input_folder =  path + r"Test - Clean speech audio flac normalized"  # Replace with your input folder path
    output_folder = path + r"Test - Noisy speech audio flac"  # Replace with your desired output folder path
    functions_NPY.delete_folder_contents(output_folder)
    P_noise_avg = functions_NPY.process_flac_files(input_folder, output_folder, noise_factor, noise_type, snr)

    # Folder to wav clean
    input_folder =  path + r"Test - Clean speech audio flac normalized"  # Replace with your input folder path
    output_folder = path + r"Test - Clean speech audio wav"  # Replace with your desired output folder path
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.convert_flac_to_wav(input_folder, output_folder)

    # Folder to wav noisy
    input_folder =  path + r"Test - Noisy speech audio flac"  # Replace with your input folder path
    output_folder = path + r"Test - Noisy speech audio wav"  # Replace with your desired output folder path
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.convert_flac_to_wav(input_folder, output_folder)

    # Split data clean
    input_folder = path + r"Test - Clean speech audio wav"
    output_folder = path + r"Test - Clean speech audio wav split"
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.split_audio_into_segments(input_folder, output_folder, segment_length)

    # Split data noisy
    input_folder = path + r"Test - Noisy speech audio wav"
    output_folder = path + r"Test - Noisy speech audio wav split"
    functions_NPY.delete_folder_contents(output_folder)
    functions_NPY.split_audio_into_segments(input_folder, output_folder, segment_length)

    print("")
    print("Done making noise")


    # Save split signals as NPY

    #  X_TRAIN SPLIT
    #Set paths to input and output data for X_TRAIN
    INPUT_DIR  = path + r"Train - Noisy speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY"
    output_name_abs = "\\X_Train_abs_split"
    output_name_phase = "\\X_Train_phase_split"
    X_Train_abs_split, X_Train_phase_split, t_stft, f_stft, scale_factor = functions_NPY.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)


    #  Y_TRAIN SPLIT
    #Set paths to input and output data for Y_TRAIN
    INPUT_DIR  = path + r"Train - Clean speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY"
    output_name_abs = "\\Y_Train_abs_split"
    output_name_phase = "\\Y_Train_phase_split"
    Y_Train_abs_split, Y_Train_phase_split, t_stft, f_stft, scale_factor = functions_NPY.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)

    
    #  X_TEST SPLIT
    #Set paths to input and output data for X_TEST
    INPUT_DIR  = path + r"Test - Noisy speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY"
    output_name_abs = "\\X_Test_abs_split"
    output_name_phase = "\\X_Test_phase_split"
    X_Test_abs_split, X_Test_phase_split, t_stft, f_stft, scale_factor = functions_NPY.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)
 


    #  X_TEST_CLEAN SPLIT
    #Set paths to input and output data for X_TEST
    INPUT_DIR  = path + r"Test - Clean speech audio wav split"
    OUTPUT_DIR = path + r"arrays_NPY"
    output_name_abs = "\\X_Test_Clean_abs_split"
    output_name_phase = "\\X_Test_Clean_phase_split"
    X_Test_Clean_abs_split, X_Test_Clean_phase_split, t_stft, f_stft, scale_factor = functions_NPY.pre_process_data(INPUT_DIR, OUTPUT_DIR, output_name_abs, output_name_phase, segment_length)


    print("Converted to NPY")

    # Print shapes

    X_Train_abs_split = np.array(X_Train_abs_split)
    Y_Train_abs_split = np.array(Y_Train_abs_split)
    X_Test_abs_split = np.array(X_Test_abs_split)
    X_Test_Clean_abs_split = np.array(X_Test_Clean_abs_split)
    print(X_Train_abs_split.shape)
    print(Y_Train_abs_split.shape)
    print(X_Test_abs_split.shape)
    print(X_Test_Clean_abs_split.shape)

    return P_noise_avg

# %%
