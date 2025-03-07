# %% EVALUATE 
import soundfile as sf
import pesq
import numpy as np
import functions
import importlib
importlib.reload(functions)


def eval(path):

    # Paths 
    #CNN_audio = path + r"\\Model_2.0 Y_Test.wav"
    ESN_audio = path + r"\ESN_model_output.wav"
    Clean_audio = path + r"\X_Test_Clean.wav"
    Test_audio = path + r"\X_Test.wav"

    # read the files
    Clean_data, Clean_samplerate = sf.read(Clean_audio) 
    Test_data, Test_samplerate = sf.read(Test_audio) 
    #CNN_data, CNN_samplerate = sf.read(CNN_audio) 
    ESN_data, ESN_samplerate = sf.read(ESN_audio) 


    # PESQ score
    freq_s = ESN_samplerate

    pesq_score_Test_Clean = pesq.pesq(freq_s, Clean_data, Clean_data, 'wb')
    print("CLEAN_PESQ_score", pesq_score_Test_Clean) 

    pesq_score_Test = pesq.pesq(freq_s, Clean_data, Test_data, 'wb')
    print("TEST_PESQ_score", pesq_score_Test)

    #pesq_score_CNN = pesq.pesq(freq_s, Clean_data, CNN_data, 'wb')
    #print("CNN_PESQ_score", pesq_score_CNN)

    pesq_score_ESN = pesq.pesq(freq_s, Clean_data, ESN_data, 'wb')
    print("ESN_PESQ_score", pesq_score_ESN)

    # SNR dummy (sum)
    sum_Test_Clean = sum(Clean_data)
    sum_Test = sum(Test_data)
    #sum_CNN_data = sum(CNN_data)
    sum_ESN_data = sum(ESN_data)

    eval = [pesq_score_Test_Clean, pesq_score_Test, pesq_score_ESN, sum_Test_Clean, sum_Test, sum_ESN_data]
    eval = np.array(eval).reshape(-1, 1)


    return eval

def eval2 (Y_pred, X_test_phase, X_test_abs, X_test_clean_abs, X_test_clean_phase, alpha, threshold):
    
    n = Y_pred.shape[0]

    pesq_score_Test = 0 
    pesq_score_Clean = 0
    pesq_score_y_pred = 0
    snr = 0
    flag = 0
    n_ignore_snr = 0

    for i in range (n): 

        # Y_pred
        input_abs = Y_pred[i]
        input_phase = X_test_phase[i]

        Y_pred_time = functions.post_process_data_array_split(input_abs, input_phase)

        #print("Y_pred time max eval" + str(abs(np.max(Y_pred_time))))

        # Test
        input_abs = X_test_abs[i]
        input_phase = X_test_phase[i]
        Test_time = functions.post_process_data_array_split(input_abs, input_phase)
        
        # Clean 
        input_abs = X_test_clean_abs[i]
        input_phase = X_test_clean_phase[i]
        Clean_time = functions.post_process_data_array_split(input_abs, input_phase)

        # 09/01 - Normalize 
        Y_pred_time = Y_pred_time/max(abs(Y_pred_time))
        Test_time = Test_time/max(abs(Test_time))
        Clean_time = Clean_time/max(abs(Clean_time))
        #print("time_sig is:" + str(Y_pred_time))
        #print("time_sig is:" + str(Test_time))
        #print("time_sig is:" + str(Clean_time))


        # PESQ
        pesq_score_Clean += pesq.pesq(16000, Clean_time, Clean_time, 'wb')
        
        pesq_score_Test += pesq.pesq(16000, Clean_time, Test_time, 'wb')
        
        pesq_score_y_pred += pesq.pesq(16000, Clean_time, Y_pred_time, 'wb')



        # SNR Out
        snr_new, flag, n_ignore = functions.SNR_of_output(Y_pred_time, Clean_time, Test_time, threshold, alpha, plot=0, flag=flag)
        #print(snr_new)
        snr += snr_new
        n_ignore_snr += n_ignore
        #print("TEST NEW!" + str(snr))


    pesq_score_Clean = pesq_score_Clean/n
    pesq_score_Test = pesq_score_Test/n
    pesq_score_y_pred = pesq_score_y_pred/n
    snr = snr/(n - n_ignore_snr)


    print("TEST !!!" + str(snr))
    #print("TEST !!!" + str(n_ignore_snr))

    return pesq_score_Clean, pesq_score_Test, pesq_score_y_pred, snr

# %%
