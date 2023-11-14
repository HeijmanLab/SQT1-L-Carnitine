# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Dr. Jordi Heijman
Date: 21/07/2023
Script: SQT1 functions
"""
#%% Import packages

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os 
import myokit

# Load the SQT1 + L-carn data
Lcarn_sqt1 = [0.017787,-48.520307,14.325950,22.303676,6.877993,0.000241,14.842432,-5.368071,-3.843856,4.941128,2.061902]

#%% Custom functions.

def export_dict_to_csv(data_dict, base_filename, steps, index = False):
    '''
    This function can be used to export the data. 

    Parameters
    ----------
    data_dict : Dictionary
        Dictionary with data.
    
    base_filename : String
        Prefix of the files.
        
    steps : List
        List of voltage steps.
        
    index : Boolean, optional.
        If you want to export the voltage steps then state True. 

    Returns
    -------
    None.

    '''
    
    for key, value in data_dict.items():
        filename = base_filename + "_" + str(key) + ".csv"
        if index is True:
            df = pd.DataFrame({'steps': steps, 'data': value})
        else: 
            df = pd.DataFrame({'data': value})
        df.to_csv(filename, index = False)

def export_dict_to_csv_AP(data_dict, base_filename):
    '''
    This function can be used to export the AP data.

    Parameters
    ----------
    data_dict : Dictionary
        Dictionary as outputted by the function 'action_pot'
    
    base_filename : String
        File name. 

    Returns
    -------
    None.

    '''
    
    filename = base_filename + ".csv"
    df = pd.DataFrame({'Time': data_dict['data']['engine.time'],
                       'vM': data_dict['data']['membrane.V'],
                       'ikr': data_dict['data']['ikr.IKr']})
    df.to_csv(filename, index = False)
    
def export_df_to_csv(steps, data, filename):
    '''
    This function can be used to export dataframes to csv.

    Parameters
    ----------
    steps : List
        List of voltage steps.
    
    data : List
        Data list.
    
    filename : String
        Name of the file. 

    Returns
    -------
    None.

    '''
    df = pd.DataFrame({'steps': steps, 'data': data})
    df.to_csv(filename + '.csv', index = False)
    
def calculate_reentry_time(i, vm, dur_sim, dur, s2, interval, cutoff):
    """
    Calculates the reentry time based on the given parameters.

    Parameters:
        i (int): Reentry iteration.
            The iteration count of the reentry process.
        vm (numpy.ndarray): Membrane potential.
            The membrane potential data.
        dur_sim (float): Duration of the S1S2 simulation.
            The total duration of the S1S2 simulation.
        dur (float): Duration of the S1 stimulus.
            The duration of the S1 stimulus.
        s2 (float): Time point for the S2 stimulus.
            The time point at which the S2 stimulus is applied.
        interval (float): Recording interval.
            The time interval between recordings.
        cutoff (float): Membrane potential regarded as inactivity.
            The threshold value below which the membrane potential is considered inactive.

    Returns:
        int: Reentry time in milliseconds.
            The duration of reentry, measured in milliseconds.
    """
    # Calculate the total duration of the reentry simulation in data points.
    iteration = i + 1
    tot_dur = ((dur_sim * iteration) / interval) + (dur / interval) - 1

    # Determine the index corresponding to the final S2 stimulus.
    final_stim = int(s2 / interval + 1)

    # Extract the membrane potential data after the final S2 stimulus.
    remain_vm = vm[final_stim:, :, :]

    # Find the maximum membrane potential after the final S2 stimulus for each recording.
    maxval = np.max(remain_vm, axis=(1, 2))

    # Find the indices where the maximum membrane potential falls below the cutoff value.
    reentry_stop_indices = np.nonzero(maxval < cutoff)[0]

    if i == 0:
        # For the first iteration, determine the reentry time based on the first occurrence of inactivity.
        time_stop = int((final_stim + reentry_stop_indices[0]) * interval)
    else:
        # For subsequent iterations, calculate the reentry time based on the last stimulus and reentry stop index.
        final_stim = int(tot_dur - (dur_sim / interval))

        if reentry_stop_indices.size > 0:
            reentry_stop = reentry_stop_indices[0]
            t = (final_stim + reentry_stop) * interval
        else:
            # If no reentry stop is found, the reentry continues until the end of the simulation.
            reentry_stop = None
            t = tot_dur * interval

        time_stop = int(t)

    # Calculate the reentry time in milliseconds.
    reentry_time_val = time_stop - s2
    
    # Print the reentry time.
    print(f'The reentry time was {reentry_time_val} ms')
    
    return reentry_time_val

def relative_apd(wt, mt, carn_wt, carn_mt):
    '''
    This function can be used to calculate the relative difference in APD
    for the different conditions.

    Parameters
    ----------
    wt : Dictionary
        WT output from 'action_potential' function
        
    mt : Dictionary
        MT output from 'action_potential' function
        
    carn_wt : Dictionary
        WT with L-Carnitine output from 'action_potential' function
        
    carn_mt : Dictionary
        MT with L-Carnitine output from 'action_potential' function

    Returns
    -------
    Dataframe with the relative changes in APD for the conditions. 

    '''
    
    rel_apd_wt = np.round((carn_wt['duration']/wt['duration'] * 100), 2)
    rel_apd_mt = np.round((carn_mt['duration']/mt['duration'] * 100), 2)
    rel_apd_NoCarn = np.round((mt['duration']/wt['duration'] * 100), 2)
    rel_apd_Carn = np.round((carn_mt['duration']/carn_wt['duration'] * 100), 2)
    
    rel_apd = [rel_apd_wt, rel_apd_mt, rel_apd_NoCarn, rel_apd_Carn]
    df = pd.DataFrame([rel_apd], columns = ['wt', 'mt', 'nocarn', 'carn'])
    
    return df