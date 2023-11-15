#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: Stefan Meier (PhD student)
Institute: CARIM, Maastricht University
Supervisor: Dr. Jordi Heijman
Date: 21/07/2023
Script: SQT1 script
"""

# Load the packages 
import matplotlib.pyplot as plt
import seaborn as sns
import myokit
import numpy as np
import pandas as pd
import os
import time
import multiprocessing

# Set your working directory.
work_dir = os.getcwd()
work_dir = os.path.join(work_dir, 'Documents', 'GitHub', 'SQT1-L-Carnitine')
os.chdir(work_dir)
files = [f for f in os.listdir('.') if os.path.isfile(f)]
for f in files:
    print(f)

# Load the functions.
from SQT1_functions import export_dict_to_csv, export_dict_to_csv_AP, export_df_to_csv, calculate_reentry_time, relative_apd

# Load the initial SQT1 parameters from the model.
x_default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
x_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1]

# Optimized parameter sets for effects of L-carnatine on SQT1.
Lcarn_sqt1 = [0.017787,-48.520307,14.325950,22.303676,6.877993,0.000241,14.842432,-5.368071,-3.843856,4.941128,2.061902]
#%% Validation correct implementation model behaviour.

# Load the Loewe et al. 2014 N588K Mutation in CRN and ORd.
m1 = myokit.load_model('MMT/ORD_LOEWE_CL_adapt_flag.mmt')
m2 = myokit.load_model('MMT/CRN_LOEWE.mmt')

# Set the cell modes to endocardial.
m1.set_value('cell.mode', 0)

# Initialize a pacing protocol.
p = myokit.Protocol()

# Set basic cycle length.
bcl = 1000

# Create an event schedule. 
p.schedule(1, 20, 3, bcl, 0)

# Compile a simulation
s1 = myokit.Simulation(m1, p)
s2 = myokit.Simulation(m2, p)

# Set the WT parameters.
s1.set_constant('ikr.mt_flag', 0)
s1.set_constant('iks.iks_scalar', 1)
s2.set_constant('ikr_loewe.mt_flag', 0)

# Set solver tolerance.
s1.set_tolerance(1e-8, 1e-8)

# Pre-pace the model.
s1.pre(bcl*1000)
s2.pre(bcl*1000)

# Run the model.
d1 = s1.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr'])
d2 = s2.run(bcl, log = ['environment.time', 'membrane.V', 'ikr_loewe.i_Kr'])

# Compile a simulation for MT.
s1MT = myokit.Simulation(m1, p)
s2MT = myokit.Simulation(m2, p)

# Set the MT parameters.
s1MT.set_constant('ikr.mt_flag', 1)
s1MT.set_constant('iks.iks_scalar', 1)
s2MT.set_constant('ikr_loewe.mt_flag', 1)

# Set solver tolerance.
s1MT.set_tolerance(1e-8, 1e-8)

# Pre-pace the model.
s1MT.pre(bcl*1000)
s2MT.pre(bcl*1000)

# Run the models again with the mutation.
d1MT = s1MT.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr'])
d2MT = s2MT.run(bcl, log = ['environment.time', 'membrane.V', 'ikr_loewe.i_Kr'])

# Plot the results.
fig, ax = plt.subplots(2, 2)
ax[0,0].plot(d1['engine.time'], d1['membrane.V'], 'k', label = 'WT')
ax[0,0].plot(d1MT['engine.time'], d1MT['membrane.V'], 'r', label = 'MT')
ax[0,0].legend()
ax[0,0].set_xlim([0, 500])
ax[0,0].set_title('ORd AP')

ax[0,1].plot(d1['engine.time'], d1['ikr.IKr'], 'k', label = 'WT')
ax[0,1].plot(d1MT['engine.time'], d1MT['ikr.IKr'], 'r', label = 'MT')
ax[0,1].legend()
ax[0,1].set_xlim([0, 500])
ax[0,1].set_title('ORd IKr')

ax[1,0].plot(d2['environment.time'], d2['membrane.V'], 'k', label = 'WT')
ax[1,0].plot(d2MT['environment.time'], d2MT['membrane.V'], 'r', label = 'MT')
ax[1,0].legend()
ax[1,0].set_xlim([0, 500])
ax[1,0].set_title('CRN AP')

ax[1,1].plot(d2['environment.time'], d2['ikr_loewe.i_Kr'], 'k', label = 'WT')
ax[1,1].plot(d2MT['environment.time'], d2MT['ikr_loewe.i_Kr'], 'r', label = 'MT')
ax[1,1].legend()
ax[1,1].set_xlim([0, 500])
ax[1,1].set_title('CRN IKr')
fig.tight_layout()

#%% Set up experimental parameters

# Load the model.
m1 = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')

# Set the cell modes to endocardial.
m1.set_value('cell.mode', 0)

# Get pacing variable.
pace = m1.get('engine.pace')

# Remove binding to pacing mechanism before voltage coupling.
pace.set_binding(None)

# Get membrane potential.
v = m1.get('membrane.V')
# Demote v from a state to an ordinary variable; no longer dynamic.
v.demote()
# right-hand side setting; value doesn't' matter because it gets linked to pacing mechanism.
v.set_rhs(0)
# Bind v's value to the pacing mechanism.
v.set_binding('pace')

# Get intracellular potassium.
ki = m1.get('potassium.Ki')
# Demote ki from a state to an ordinary variable; no longer dynamic.
ki.demote()
# Set rhs to 120 according to Odening et al. 2019.
ki.set_rhs(120)

#%% Voltage-steps protocol

## Initialize the steps in patch-clamp protocol and add a small margin due to problems 
## w/ zero division.
steps_act = np.arange(-31, 60, 10) 
p_act = myokit.Protocol()
for k,step in enumerate(steps_act):
    # 1500ms of holding potential
    p_act.add_step(-40, 1500) 
    # Voltage step for 1500 ms
    p_act.add_step(step, 1500)
    # 2000 ms repolarizing step for tail current
    p_act.add_step(-30, 2000) 
    # resume holding potential for 4500ms
    p_act.add_step(-40, 4500)
t_act = p_act.characteristic_time()-1
s = myokit.Simulation(m1, p_act)
#%% Parameter optimization to minimize the sum of squared errors between the model and the experimental behaviour

def err_func(x, xfull, inds, carn_steady, carn_tail, steady_sqt, tail_sqt,
             mt_flag = False, norm = False, showit = False, showerror = True, return_err = True,
             carn_flag = False):
    """
    Parameter optimization to minimize the sum of squared errors between the model and the experimental behavior.
    
    This function performs parameter optimization to minimize the sum of squared errors between the model and the experimental behavior of the action potential. The action potential simulation is based on the provided model and protocol. The optimization aims to fit the model's steady-state and tail currents for IKr and IKs to experimental data.
    
    Parameters:
    ----------
    x : list of float
        A list of parameter values for the iKr (rapid delayed rectifier potassium current) model components to be optimized.
        
    xfull : list of float
        The full set of parameters for the iKr model. This list contains both the optimized parameters and constant parameters.
        
    inds : list of int
        A list of indices specifying the positions of the parameters to be optimized within the 'xfull' list.
        
    carn_steady : list of float
        Experimental steady-state IKr currents obtained after L-carnitine treatment.
        
    carn_tail : list of float
        Experimental tail IKr currents obtained after L-carnitine treatment.
        
    steady_sqt : list of float
        Reference steady IKr currents obtained from simulation using default parameters (sqt1).
        
    tail_sqt : list of float
        Reference tail IKr currents obtained from simulation using default parameters (sqt1).
        
    mt_flag : bool, optional (default = False)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'mt' condition.
        
    norm : bool, optional (default = False)
        Flag indicating whether the model and experimental data should be normalized for plotting.
        
    showit : bool, optional (default = False)
        Flag indicating whether to display plots of the model and experimental data for visual comparison.
        
    showerror : bool, optional (default = True)
        Flag indicating whether to display the calculated error values during the optimization process.
        
    return_err : bool, optional (default = True)
        Flag indicating whether to return the final calculated error after optimization or additional data for analysis.
    
    carn_flag : bool, optional (default = False)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'carn' condition.
    
    Returns:
    -------
    float or dict
        If 'return_err' is True, the function returns the final calculated error value after optimization.
        If 'return_err' is False, the function returns a dictionary containing the following elements:
        - 'steps': The voltage steps used in the protocol.
        - 'ikr_steady': Model steady IKr currents obtained after simulation.
        - 'ikr_tail': Model tail IKr currents obtained after simulation.
        - 'iks_steady': Model steady IKs currents obtained after simulation.
        - 'iks_tail': Model tail IKs currents obtained after simulation.
        - 'ref_steady': Reference steady IKr currents obtained from simulation with default parameters.
        - 'ref_tail': Reference tail IKr currents obtained from simulation with default parameters.
    
    Note:
    -----
    - The function uses the 'err_func' to calculate the errors and optimize the parameters for the iKr model.
    - The CVODE solver tolerance is set to 1e-8 for numerical stability during simulation.
    - The function uses the 's' simulation object, which must be set up before calling this function.
    - The experimental data and reference data should be provided for comparison during the optimization process.
    - The 'inds' parameter allows for selecting specific parameters for optimization, while keeping others constant.
    - The optimization aims to fit the steady-state and tail IKr currents to experimental data.
    - The function can also visualize the fitted model and experimental data for comparison, as well as display error values.
    - Use 'norm=True' to normalize the data for plotting to compare relative behavior.
    """
    
    # Reset to initial states and re-run simulation for every iteration of the error function.
    s.reset()
    
    # Re-initiate the protocol.
    s.set_protocol(p_act)
    
    # Set the maximum stepsize to 2ms to obtain better step curves.
    s.set_max_step_size(2) 
     
    # Set the extracellular potassium concentration to Odening et al. 2019.
    s.set_constant('extra.Ko', 5.4)

    # Set tolerance to counter suboptimal optimalisation with CVODE.
    s.set_tolerance(1e-8, 1e-8)

    # Set the indices 
    xarr = np.array(xfull)
    xarr[inds] = np.array(x)
    
    # Set the parameters
    for i in range(len(xarr)):
        s.set_constant(f"ikr.p{i+1}", xarr[i])
    
    if mt_flag is False:
        if carn_flag is False: 
            s.set_constant('iks.iks_scalar', 1)
        else:
            s.set_constant('iks.iks_scalar', 0.75)
    else:
        if carn_flag is False:
            s.set_constant('iks.iks_scalar', 1.35)
        else: 
            s.set_constant('iks.iks_scalar', 0.88)

    # Run the simulation protocol and log several variables.
    d = s.run(t_act)
    
    # Plot the curves for each step.
    if showit is True: 
        plt.figure()
        plt.subplot(3, 1, 1)
        plt.plot(d['engine.time'], d['membrane.V'])
        plt.xlabel('Time [ms]')
        plt.ylabel('vM [mV]')
        plt.title('Voltage-clamp protocol in steps')
        plt.subplot(3, 1, 2)
        plt.plot(d['engine.time'], d['ikr.IKr'])  
        plt.title('IKr traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.subplot(3, 1, 3)
        plt.plot(d['engine.time'], d['iks.IKs'])
        plt.title('IKs traces')
        plt.xlabel('Time [ms]')
        plt.ylabel('pA/pF')
        plt.tight_layout()
        
    # Split the log into smaller chunks to overlay; to get the individual steps.
    ds = d.split_periodic(9500, adjust=True) 

    # Initiate the peak current variable.
    Ikr_steady = np.zeros(len(ds)) 
    Iks_steady = np.zeros(len(ds))

    # Initiate the tail current variable.
    Ikr_tail = np.zeros(len(ds)) 
    Iks_tail = np.zeros(len(ds))
    
    # Trim each new log to contain the steps of interest by enumerating through the individual duration steps.
    for k, d in enumerate(ds):
        # Adjust is the time at the start of every sweep which is set to zero.
        steady = d.trim_left(1501, adjust = True) 
        
        # Duration of the peak/steady current, shorter than max duration to prevent interference between steady peak and upslope of tail.
        steady = steady.trim_right(1498) 
        
        # Total step duration (holding potential + peak current + margin of 1ms) to ensure the tail current.
        tail = d.trim_left(3001, adjust = True) 
        
        # Duration of the tail current.
        tail = tail.trim_right(2000) 
        
        # Obtain the max of the steady. 
        Ikr_steady[k] = max(steady['ikr.IKr'])
        Iks_steady[k] = max(steady['iks.IKs']) 
        
        # Obtain the max of the tail.
        Ikr_tail[k] = max(tail['ikr.IKr']) 
        Iks_tail[k] = max(tail['iks.IKs']) 

    # Calculate the ratio of between the model tail and the MT tail, vice versa for steady.
    if mt_flag is True: 
        Ikr_tail_ratio = Ikr_tail[4:9]/tail_sqt[4:9]
        Ikr_steady_ratio = Ikr_steady[5:10]/steady_sqt[5:10]
    else:
        Ikr_tail_ratio = Ikr_tail[5:10]/tail_sqt[5:10]
        Ikr_steady_ratio = Ikr_steady[0:10]/steady_sqt[0:10]
        
    # Plot the Steady and Tail.
    if showit is True:
        plt.figure()
        plt.subplot(1,3,1)
        plt.plot(steps_act, steady_sqt, 'k', label = 'Reference steady')
        plt.plot(steps_act, tail_sqt, 'r', label = 'Reference tail')
        plt.plot(steps_act, Ikr_steady, 'k', ls = 'dotted', label = 'Fit steady')
        plt.plot(steps_act, Ikr_tail, 'r', ls = 'dotted', label = 'Fit tail')
        plt.title('Steady and tail IKr currents')
        plt.ylabel('pA/pF')
        plt.xlabel('Time [ms]')
        plt.legend()
        
        plt.subplot(1,3,2)
        
        plt.plot(steps_act, Iks_steady, 'k', ls = 'dotted', label = 'Fit steady')
        plt.plot(steps_act, Iks_tail, 'r', ls = 'dotted', label = 'Fit tail')
        plt.title('Steady and tail IKs currents')
        plt.ylabel('pA/pF')
        plt.xlabel('Time [ms]')
        plt.legend()
        plt.tight_layout()

        plt.figure()
        if mt_flag is True:
            plt.plot(steps_act[4:9], Ikr_tail_ratio, 'k')
            plt.plot(steps_act[5:10], Ikr_steady_ratio, 'r')
        else:
            plt.plot(steps_act[5:10], Ikr_tail_ratio, 'k')
            plt.plot(steps_act[0:10], Ikr_steady_ratio, 'r')

    if norm is True:
        # Normalize the model and experimental data.
        sns_model = Ikr_steady/max(Ikr_steady)
        tns_model =  Ikr_tail/max(Ikr_steady)
        tnt_model = Ikr_tail/max(Ikr_tail)
        
        # Normalize the experimental data.
        sns_data = [i/max(carn_steady) for i in carn_steady]
        tns_data = [i/max(carn_steady) for i in carn_tail]
        tnt_data = [i/max(carn_tail) for i in carn_tail]
        
        if showit is True:
            fig, ax = plt.subplots(1, 3)
            
            ax[0].set_title('SnS')
            ax[0].plot(steps_act, sns_model, label = "Model", color = 'blue')
            ax[0].plot(steps_act, sns_data, 'o', label ='Carnitine', color = 'orange')
            ax[0].set_ylabel('I/Imax')
            ax[0].set_xlabel('Prepulse potential [mV]')
            ax[0].legend(loc = 'upper left')
            
            ax[1].set_title('TnS')
            ax[1].plot(steps_act, tns_model, label = "Model", color = 'blue')
            ax[1].plot(steps_act, tns_data, 'o', label ='Carnitine', color = 'orange')
            ax[1].set_ylabel('I/Imax')
            ax[1].set_xlabel('Prepulse potential [mV]')
            ax[1].legend(loc = 'upper left')
            
            ax[2].set_title('Tnt')
            ax[2].plot(steps_act, tnt_model, label = "Model", color = 'blue')
            ax[2].plot(steps_act, tnt_data, 'o', label ='Carnitine', color = 'orange')
            ax[2].set_ylabel('I/Imax')
            ax[2].set_xlabel('Prepulse potential [mV]')
            ax[2].legend(loc = 'upper left')
            fig.tight_layout()
    
    if mt_flag is True:
        error_tail = sum((Ikr_tail_ratio - 0.55)**2)
        error_steady = sum((Ikr_steady_ratio - 0.75)**2)
    else: 
        error_tail = sum((Ikr_tail_ratio - 0.9)**2)
        error_steady = sum((Ikr_steady_ratio - 1) **2)
   
    # Calculate the sum of squared errors for the sign changes
    if mt_flag is True:
        x_default = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
    else:
        x_default = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1]
    error_signchange = sum([5 if (i < 0 and j > 0) or (i > 0 and j < 0) else 0 for i, j in zip(x_default, xarr)])
   
    # Add the errors
    error = error_tail + error_steady + error_signchange
    
    # Print the error terms
    if showerror is True:
        print ("X = [%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f] # Error: %f, %f, %f, %f"% (xarr[0], xarr[1], xarr[2], 
                                                                                  xarr[3], xarr[4], xarr[5], 
                                                                                  xarr[6], xarr[7], xarr[8], 
                                                                                  xarr[9], xarr[10], error,
                                                                                  error_tail, error_steady, 
                                                                                 error_signchange)) 
    if return_err is True:
        return error
    else:
        return dict(steps = steps_act, ikr_steady = Ikr_steady, ikr_tail = Ikr_tail, iks_steady = Iks_steady, iks_tail = Iks_tail,
                    ref_steady = steady_sqt, ref_tail = tail_sqt)
#%% Perform the error minimization

# Load the L-Carnitine treatment data on IKr.
ikr_carnL_steady = [0.3636, 0.5519, 0.7299, 0.7121, 0.6946, 0.5257, 0.5198, 0.5893, 0.7299, 0.7713]
ikr_carnL_tail = [-0.027, -0.0489, -0.0243, -0.0569, 0.0248, 0.0278, 0.1121, 0.1385, 0.0537, 0.0962]

# Reference tail and steady after running with default parameters once.
ikr_tail_sqt1 = [0.18019632, 0.50991302, 1.04082499, 1.36457873, 1.46874907, 1.49463627, 1.50069027, 1.50207747, 1.50239432, 1.50246672]
ikr_steady_sqt1 = [0.15521109, 0.58902343, 1.35735324, 1.9338675,  2.16788196, 2.18397432, 2.04187431, 1.77514567, 1.43494635, 1.0848202]
ikr_tail_wt = [0.08188738, 0.25800685, 0.62721595, 0.90409909, 0.99680298, 1.01904937, 1.02396182, 1.02502317, 1.02525154, 1.02530066]
ikr_steady_wt = [0.06987978, 0.25910039, 0.58859401, 0.7399551, 0.66995722, 0.53518835, 0.4040717, 0.29504964, 0.21059247, 0.14788093]

# Load the experimental IKs data.
iks_tail_exp= [0.00031467, 0.00278388, 0.01365995, 0.03223178, 0.04745116, 0.05545458, 0.05876223, 0.06000359, 0.06050124, 0.06080238]
iks_steady_exp = [4.62992631e-04, 4.51500588e-03, 2.73979567e-02, 1.02117585e-01, 2.23041986e-01, 3.25602655e-01, 3.94253134e-01, 4.45406758e-01, 4.90144289e-01, 5.33147829e-01]

# Load the initial SQT1 parameters from the model.
x_default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
x_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1]

# Optimized parameter sets for effects of L-carnatine on SQT1 and WT.
Lcarn_sqt1 = [0.017787,-48.520307,14.325950,22.303676,6.877993,0.000241,14.842432,-5.368071,-3.843856,4.941128,2.061902]
Lcarn_wt = [0.024391,8.920430,21.917451,14.934982,6.413404,0.000321,18.904892,-5.269330,-3.220124,4.772577,2.252838]

# Initialize the indices.
inds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 

# Select the WT data.
xfull_WT = np.array(x_wt)
xsel_WT = xfull_WT[inds]

# Run the simulation for the WT. 
Odening_WT = err_func(xsel_WT, xfull = xfull_WT, inds = inds, carn_steady = ikr_carnL_steady, carn_tail = ikr_carnL_tail, 
                   steady_sqt = ikr_steady_wt, tail_sqt = ikr_tail_wt, mt_flag = False, norm = False, 
                   showit = True, showerror = True, return_err = False, carn_flag = False)    

# Select the MT data.
xfull_MT = np.array(x_default_sqt)
xsel_MT = xfull_MT[inds]

# Run the simulation for the MT.
Odening_MT = err_func(xsel_MT, xfull = xfull_MT, inds = inds, carn_steady = ikr_carnL_steady, carn_tail = ikr_carnL_tail, 
                   steady_sqt = ikr_steady_sqt1, tail_sqt = ikr_tail_sqt1, mt_flag = True, norm = False, 
                   showit = True, showerror = True, return_err = False, carn_flag = False)    

# Select the L-carnitine WT data.
xfull_carn_WT = np.array(Lcarn_wt)
xsel_carn_WT = xfull_carn_WT[inds]

# Run the simulation for the L-carnitine WT data.
Odening_WT_carn = err_func(xsel_carn_WT, xfull = xfull_carn_WT, inds = inds, carn_steady = ikr_carnL_steady, carn_tail = ikr_carnL_tail, 
                   steady_sqt = ikr_steady_wt, tail_sqt = ikr_tail_wt, mt_flag = False, norm = False, 
                   showit = True, showerror = True, return_err = False, carn_flag = True)    

# Select the Carnitine data. 
xfull_carn = np.array(Lcarn_sqt1)
xsel_carn = xfull_carn[inds]

# Run the simulation for the L-carnitine SQT1 data.
Odening_MT_carn = err_func(xsel_carn, xfull = xfull_carn, inds = inds, carn_steady = ikr_carnL_steady, carn_tail = ikr_carnL_tail, 
                   steady_sqt = ikr_steady_sqt1, tail_sqt = ikr_tail_sqt1, mt_flag = True, norm = False, 
                   showit = True, showerror = True, return_err = False, carn_flag = True)    

# Validate the L-carnitine effects.
steady_WTc_model = Odening_WT_carn['ikr_steady'] / Odening_WT['ikr_steady']
tail_WTc_model = Odening_WT_carn['ikr_tail'] / Odening_WT['ikr_tail']

steady_MTc_model = Odening_MT_carn['ikr_steady'] / Odening_MT['ikr_steady']
tail_MTc_model = Odening_MT_carn['ikr_tail'] / Odening_MT['ikr_tail']

# Define the experimental difference between WT or MT and L-carnitine
steady_WTc_exp = [0.906128161, 0.967050288, 0.968234228, 0.990470163, 0.991601229, 0.970878784, 0.947880997, 0.902929898, 0.860902054, 0.765018895]
tail_WTc_exp = [0.441092864, 0.590463167, -0.322282419, 0.223993477, 0.827774321, 1.171965435, 0.931265989, 0.913194664, 1.073569586, 0.939633078]

steady_MTc_exp = [0.999096421, 0.926712674, 0.986305557, 0.914740211, 0.926140203, 0.763546494, 0.722983535, 0.730808936, 0.760772214, 0.710300627]
tail_MTc_exp = [2.226311686, 0.972059417, 0.618583965, 0.942812407, 0.537714113, 0.316838893, 0.741925029, 0.692385396, 0.498884765, 0.847393936]

# Select the correct voltage range (tail = +10 - +50mV, steady = +20 mV to +60 mV)
steady_WTc_exp_sel = steady_WTc_exp[5:10]
tail_WTc_exp_sel = tail_WTc_exp[4:9]
steady_WTc_model_sel = steady_WTc_model[5:10]
tail_WTc_model_sel = tail_WTc_model[4:9]

steady_MTc_exp_sel = steady_MTc_exp[5:10]
tail_MTc_exp_sel = tail_MTc_exp[4:9]
steady_MTc_model_sel = steady_MTc_model[5:10]
tail_MTc_model_sel = tail_MTc_model[4:9]

# Define the range from steps_act.
steady_steps = steps_act[5:10]
tail_steps = steps_act[4:9]

# Export to Graphpad.
export_dict_to_csv(Odening_WT, base_filename = 'Odening_WT', steps = steps_act)
export_dict_to_csv(Odening_MT, base_filename = 'Odening_MT', steps = steps_act)
export_dict_to_csv(Odening_WT_carn, base_filename = 'Odening_WT_carn', steps = steps_act)
export_dict_to_csv(Odening_MT_carn, base_filename = 'Odening_MT_carn', steps = steps_act)
    
export_df_to_csv(steps = steady_steps, data = steady_WTc_exp_sel, filename = 'steady_WTc_exp')
export_df_to_csv(steps = tail_steps, data = tail_WTc_exp_sel, filename = 'tail_WTc_exp')
export_df_to_csv(steps = steady_steps, data = steady_MTc_exp_sel, filename = 'steady_MTc_exp')
export_df_to_csv(steps = tail_steps, data = tail_MTc_exp_sel, filename = 'tail_MTc_exp')

export_df_to_csv(steps = steady_steps, data = steady_WTc_model_sel, filename = 'steady_WTc_model')
export_df_to_csv(steps = tail_steps, data = tail_WTc_model_sel, filename = 'tail_WTc_model')
export_df_to_csv(steps = steady_steps, data = steady_MTc_model_sel, filename = 'steady_MTc_model')
export_df_to_csv(steps = tail_steps, data = tail_MTc_model_sel, filename = 'tail_MTc_model')

#%% Action potential effects

# Load the model.
m1 = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
m2 = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
m3 = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')

# Set cell type.
m1.set_value('cell.mode', 0) # Endo
m2.set_value('cell.mode', 1) # Epi
m3.set_value('cell.mode', 2) # Mid

# Create an action potential protocol.
pace = myokit.Protocol()

# Set the basic cycle length to 1 Hz.
bcl = 1000

# Create an event schedule.
pace.schedule(1, 20, 0.5, bcl)

def action_pot(m, p, x, bcl, prepace, mt_flag = True, carn_flag = False):
    """
    Action potential effects
    
    This script performs a simulation of action potential effects using a given model and action potential protocol. It calculates the action potential duration (APD) and other relevant data based on the specified parameters.
    
    Parameters:
    ----------
    m : myokit.Model
        The Myokit model representing the cellular electrophysiology.
    
    p : myokit.Protocol
        The action potential protocol to be applied during the simulation.
    
    x : list of float
        List of parameter values to be set for the iKr (rapid delayed rectifier potassium current) model components.
    
    bcl : int
        The basic cycle length in milliseconds (ms) used in the action potential protocol.
    
    prepace : int
        The number of pre-pace cycles to stabilize the model before starting the simulation.
    
    
    mt_flag : bool, optional (default = True)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'mt' condition.
    
    carn_flag : bool, optional (default = False)
        Flag indicating whether the simulation should include a modification to the iKr current based on the 'carn' condition.
    
    Returns:
    -------
    dict
        A dictionary containing the following elements:
        - 'data': The simulation data, including time, membrane potential (V), and iKr current (IKr).
        - 'apd': A myokit.APDMeasurement object representing the action potential duration data.
        - 'duration': The calculated action potential duration (APD90) in milliseconds (ms).
        - 'ikr': The iKr current data obtained from the simulation.
    
    Note:
    -----
    - The simulation will adjust the iKr current based on the provided 'mt_flag' and 'carn_flag' conditions.
    - The CVODE solver tolerance is set to 1e-8 for numerical stability.
    - The action potential is pre-paced for a specified number of cycles ('prepace') to stabilize the model.
    - The action potential duration (APD) is calculated using a threshold of 90% repolarization (APD90).
    """


    # Create a simulation object.
    sim = myokit.Simulation(m, p)
    
    # Set CVODE solver tolerance.
    sim.set_tolerance(1e-8, 1e-8)
    
    # Set the parameters.
    for i in range(len(x)):
        sim.set_constant(f"ikr.p{i+1}", x[i])
    
    if mt_flag is False:
        if carn_flag is False: 
            sim.set_constant('iks.iks_scalar', 1)
        else:
            sim.set_constant('iks.iks_scalar', 0.75)
    else:
        if carn_flag is False:
            sim.set_constant('iks.iks_scalar', 1.35)
        else: 
            sim.set_constant('iks.iks_scalar', 0.88)
    
    # Pre-pace the model.
    sim.pre(prepace * bcl)
    
    # Run the simulation and calculate the APD90.
    vt = 0.9 * sim.state()[m.get('membrane.V').indice()]
    data, apd = sim.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr'], apd_variable = 'membrane.V', apd_threshold = vt)
    
    # Get IKr out of the simulation.
    ikr = data['ikr.IKr']
    
    # Determine the APD duration.
    duration = round(apd['duration'][0], 2)    
    
    return dict(data = data, apd = apd, duration = duration, ikr = ikr)

# Generate action potentials for each cell type
wt_ap_endo = action_pot(m = m1, p = pace, x = x_wt, bcl = bcl, prepace = 1000, mt_flag = False)
Lcarn_wt_ap_endo = action_pot(m = m1, p = pace, x = Lcarn_wt, bcl = bcl, prepace = 1000, mt_flag = False)
sqt_ap_endo = action_pot(m = m1, p = pace, x = x_default_sqt, bcl = bcl, prepace = 1000, mt_flag = True)
Lcarn_sqt_ap_endo = action_pot(m = m1, p = pace, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, mt_flag = True)

wt_ap_epi = action_pot(m = m2, p = pace, x = x_wt, bcl = bcl, prepace = 1000, mt_flag = False)
Lcarn_wt_ap_epi = action_pot(m = m2, p = pace, x = Lcarn_wt, bcl = bcl, prepace = 1000, mt_flag = False)
sqt_ap_epi = action_pot(m = m2, p = pace, x = x_default_sqt, bcl = bcl, prepace = 1000, mt_flag = True)
Lcarn_sqt_ap_epi = action_pot(m = m2, p = pace, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, mt_flag = True)

wt_ap_mid = action_pot(m = m3, p = pace, x = x_wt, bcl = bcl, prepace = 1000, mt_flag = False)
Lcarn_wt_ap_mid = action_pot(m = m3, p = pace, x = Lcarn_wt, bcl = bcl, prepace = 1000, mt_flag = False)
sqt_ap_mid = action_pot(m = m3, p = pace, x = x_default_sqt, bcl = bcl, prepace = 1000, mt_flag = True)
Lcarn_sqt_ap_mid = action_pot(m = m3, p = pace, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, mt_flag = True)

# Set up the figure with 3 rows and 2 columns
fig, axs = plt.subplots(3, 2, figsize=(12, 18))

# Plot the results for endo
axs[0, 0].plot(wt_ap_endo['data']['engine.time'], wt_ap_endo['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {wt_ap_endo['duration']} ms")
axs[0, 0].plot(Lcarn_wt_ap_endo['data']['engine.time'], Lcarn_wt_ap_endo['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_wt_ap_endo['duration']} ms")
axs[0, 0].legend()
axs[0, 0].set_ylabel('Membrane potential [mV]')
axs[0, 0].set_xlabel('Time [ms]')
axs[0, 0].set_title('WT Loewe model (Endo)')
axs[0, 0].set_xlim([0, 500])

axs[0, 1].plot(sqt_ap_endo['data']['engine.time'], sqt_ap_endo['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {sqt_ap_endo['duration']} ms")
axs[0, 1].plot(Lcarn_sqt_ap_endo['data']['engine.time'], Lcarn_sqt_ap_endo['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_sqt_ap_endo['duration']} ms")
axs[0, 1].legend()
axs[0, 1].set_ylabel('Membrane potential [mV]')
axs[0, 1].set_xlabel('Time [ms]')
axs[0, 1].set_title('SQT1 Loewe model (Endo)')
axs[0, 1].set_xlim([0, 500])

# Plot the results for epi
axs[1, 0].plot(wt_ap_epi['data']['engine.time'], wt_ap_epi['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {wt_ap_epi['duration']} ms")
axs[1, 0].plot(Lcarn_wt_ap_epi['data']['engine.time'], Lcarn_wt_ap_epi['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_wt_ap_epi['duration']} ms")
axs[1, 0].legend()
axs[1, 0].set_ylabel('Membrane potential [mV]')
axs[1, 0].set_xlabel('Time [ms]')
axs[1, 0].set_title('WT Loewe model (Epi)')
axs[1, 0].set_xlim([0, 500])

axs[1, 1].plot(sqt_ap_epi['data']['engine.time'], sqt_ap_epi['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {sqt_ap_epi['duration']} ms")
axs[1, 1].plot(Lcarn_sqt_ap_epi['data']['engine.time'], Lcarn_sqt_ap_epi['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_sqt_ap_epi['duration']} ms")
axs[1, 1].legend()
axs[1, 1].set_ylabel('Membrane potential [mV]')
axs[1, 1].set_xlabel('Time [ms]')
axs[1, 1].set_title('SQT1 Loewe model (Epi)')
axs[1, 1].set_xlim([0, 500])

# Plot the results for mid
axs[2, 0].plot(wt_ap_mid['data']['engine.time'], wt_ap_mid['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {wt_ap_mid['duration']} ms")
axs[2, 0].plot(Lcarn_wt_ap_mid['data']['engine.time'], Lcarn_wt_ap_mid['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_wt_ap_mid['duration']} ms")
axs[2, 0].legend()
axs[2, 0].set_ylabel('Membrane potential [mV]')
axs[2, 0].set_xlabel('Time [ms]')
axs[2, 0].set_title('WT Loewe model (Mid)')
axs[2, 0].set_xlim([0, 500])

axs[2, 1].plot(sqt_ap_mid['data']['engine.time'], sqt_ap_mid['data']['membrane.V'], 'k', label=f"No L-carnitine, APD = {sqt_ap_mid['duration']} ms")
axs[2, 1].plot(Lcarn_sqt_ap_mid['data']['engine.time'], Lcarn_sqt_ap_mid['data']['membrane.V'], 'r', label=f"With L-carnitine, APD = {Lcarn_sqt_ap_mid['duration']} ms")
axs[2, 1].legend()
axs[2, 1].set_ylabel('Membrane potential [mV]')
axs[2, 1].set_xlabel('Time [ms]')
axs[2, 1].set_title('SQT1 Loewe model (Mid)')
axs[2, 1].set_xlim([0, 500])

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()

# Export to GraphPad.
export_dict_to_csv_AP(wt_ap_endo, base_filename = 'AP_WT_endo')
export_dict_to_csv_AP(Lcarn_wt_ap_endo, base_filename = 'AP_WTCarn_endo')
export_dict_to_csv_AP(sqt_ap_endo, base_filename = 'AP_MT_endo')
export_dict_to_csv_AP(Lcarn_sqt_ap_endo, base_filename = 'AP_MTCarn_endo')

export_dict_to_csv_AP(wt_ap_epi, base_filename = 'AP_WT_epi')
export_dict_to_csv_AP(Lcarn_wt_ap_epi, base_filename = 'AP_WTCarn_epi')
export_dict_to_csv_AP(sqt_ap_epi, base_filename = 'AP_MT_epi')
export_dict_to_csv_AP(Lcarn_sqt_ap_epi, base_filename = 'AP_MTCarn_epi')

export_dict_to_csv_AP(wt_ap_mid, base_filename = 'AP_WT_mid')
export_dict_to_csv_AP(Lcarn_wt_ap_mid, base_filename = 'AP_WTCarn_mid')
export_dict_to_csv_AP(sqt_ap_mid, base_filename = 'AP_MT_mid')
export_dict_to_csv_AP(Lcarn_sqt_ap_mid, base_filename = 'AP_MTCarn_mid')

# Obtain the relative changes in apd for each of the celltypes. 
rel_apd_endo = relative_apd(wt = wt_ap_endo, mt = sqt_ap_endo, 
                      carn_wt = Lcarn_wt_ap_endo, carn_mt = Lcarn_sqt_ap_endo)

rel_apd_epi = relative_apd(wt = wt_ap_epi, mt = sqt_ap_epi, 
                      carn_wt = Lcarn_wt_ap_epi, carn_mt = Lcarn_sqt_ap_epi)

rel_apd_mid = relative_apd(wt = wt_ap_mid, mt = sqt_ap_mid, 
                      carn_wt = Lcarn_wt_ap_mid, carn_mt = Lcarn_sqt_ap_mid)

# Relabel the x-axis.
rel_apd_labels = ['WT + L-Carn / WT', 'SQT1 + L-Carn / SQT1',
                  'SQT1 / WT', 'SQT1 + L-Carn / WT + L-Carn']

# Visualize the relative difference in APD.
fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (8, 12))

# Plot bar plots for endo.
sns.barplot(x = rel_apd_endo.columns, y = rel_apd_endo.iloc[0], ax = axes[0], color = 'black')
axes[0].set_title('Endo', fontweight = 'bold')
axes[0].set_ylabel('Relative change in APD (%)')
axes[0].set_xticklabels(rel_apd_labels)

# Plot bar plots for epi.
sns.barplot(x = rel_apd_epi.columns, y = rel_apd_epi.iloc[0], ax = axes[1], color = 'darkgrey')
axes[1].set_title('Epi', fontweight = 'bold')
axes[1].set_ylabel('Relative change in APD (%)')
axes[1].set_xticklabels(rel_apd_labels)

# Plot bar plots for mid.
sns.barplot(x = rel_apd_mid.columns, y = rel_apd_mid.iloc[0], ax = axes[2], color = 'lightgrey')
axes[2].set_title('Mid', fontweight = 'bold')
axes[2].set_ylabel('Relative change in APD (%)')
axes[2].set_xticklabels(rel_apd_labels)

# Adjust layout.
plt.tight_layout()

# Show the plots.
plt.show()
#%% Rabbit ventricular model

# Load the rabbit model
mr = myokit.load_model('MMT/Mahajan-2008.mmt')

# Create an action potential protocol.
pr = myokit.Protocol()

# Set the basic cycle length to 1 Hz.
bcl = 1000

# Create an event schedule.
pr.schedule(1, 20, 0.5, bcl)

# Create a simulation object.
sim_r = myokit.Simulation(mr, pr)

# Set CVODE solver tolerance.
sim_r.set_tolerance(1e-8, 1e-8)

# Set the IKr modeltype to use
sim_r.set_constant('IKr.IKr_modeltype', 0)

# Pre-pace the model
sim_r.pre(bcl * 1000)

# Run the simulation and calculate the APD90.
vt_r = 0.9 * sim_r.state()[mr.get('cell.V').index()]
data_r, apd_r = sim_r.run(bcl, log = ['Environment.time', 'cell.V', 'IKr.xikr'], apd_variable = 'cell.V', apd_threshold = vt_r)

# Round the APD90.
apd90_r = round(apd_r['duration'][0], 2)    

# Reset the simulation and re-run for the Loewe et al. (2014) IKr MM formulation
sim_l = myokit.Simulation(mr, pr)

# Set CVODE solver tolerance.
sim_l.set_tolerance(1e-8, 1e-8)

# Set the IKr modeltype to use
sim_l.set_constant('IKr.IKr_modeltype', 1)

# If modeltype is 1, then scale IKr from Loewe to match the rabbit one.
sim_l.set_constant('IKr_MM.IKr_scalar', 0.3)

# Pre-pace the model
sim_l.pre(bcl * 1000)

# Run the simulation and calculate the APD90.
vt_l = 0.9 * sim_l.state()[mr.get('cell.V').index()]
data_l, apd_l = sim_l.run(bcl, log = ['Environment.time', 'cell.V', 'IKr.xikr'], apd_variable = 'cell.V', apd_threshold = vt_l)

# Round the APD90.
apd90_l = round(apd_l['duration'][0], 2)    

# Visualize the results
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

axs[0].plot(data_r['Environment.time'], data_r['cell.V'], 'k', label = f'Mahajan 2008, apd90 = {apd90_r} ms')
axs[0].plot(data_l['Environment.time'], data_l['cell.V'], 'r', label = f'Loewe 2014, apd90 = {apd90_l} ms')
axs[0].legend()
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane potential (mV)')

axs[1].plot(data_r['Environment.time'], data_r['IKr.xikr'], 'k', label = f'Mahajan 2008, apd90 = {apd90_r} ms')
axs[1].plot(data_l['Environment.time'], data_l['IKr.xikr'], 'r', label = f'Loewe 2014, apd90 = {apd90_l} ms')
axs[1].legend()
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Current density (pA/pF)')

# Tidy up the plots
plt.tight_layout()
plt.show()

#%% Rabbit model APs and effects of MT and L-Carn treatment

# Simulate the effects of the MT and the L-Carnitine treatment in the rabbit model
# with the Loewe et al. (2014) Markov Model.

# Load the rabbit model
mrl = myokit.load_model('MMT/Mahajan-2008.mmt')

# Create an action potential protocol.
prl = myokit.Protocol()

# Create an event schedule.
prl.schedule(1, 20, 0.5, bcl)

def rabbit_loewe(mrl, prl, x, prepace):

    # Create a simulation object.
    sim_rl = myokit.Simulation(mrl, prl)
    
    # Set CVODE solver tolerance.
    sim_rl.set_tolerance(1e-8, 1e-8)
    
    # Set the IKr modeltype to Loewe
    sim_rl.set_constant('IKr.IKr_modeltype', 1)
    
    # Scale IKr formulation of Loewe down to 30%
    sim_rl.set_constant('IKr_MM.IKr_scalar', 0.3)
    
    # Set the other model parameters to Loewe as well
    for i in range(len(x)):
        sim_rl.set_constant(f'IKr_MM.p{i+1}', x[i])
    
    # Pre-pace the model
    sim_rl.pre(bcl * prepace)
    
    # Run the simulation and calculate the APD90.
    vt_rl = 0.9 * sim_rl.state()[mrl.get('cell.V').index()]
    data_rl, apd_rl = sim_rl.run(bcl, log = ['Environment.time', 'cell.V', 'IKr.xikr'], apd_variable = 'cell.V', apd_threshold = vt_rl)
    
    # Round the APD90.
    apd90_rl = round(apd_rl['duration'][0], 2)    
    
    return dict(data = data_rl, apd = apd90_rl)

RL_wt = rabbit_loewe(mrl = mrl, prl = prl, x = x_wt, prepace = 1000)
RL_sqt1 = rabbit_loewe(mrl = mrl, prl = prl, x = x_default_sqt, prepace = 1000)
RL_Lcarn_wt = rabbit_loewe(mrl = mrl, prl = prl, x = Lcarn_wt, prepace = 1000)
RL_Lcarn_sqt1 = rabbit_loewe(mrl = mrl, prl = prl, x = Lcarn_sqt1, prepace = 1000)
 
# Visualize the results
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

axs[0, 0].plot(RL_wt['data']['Environment.time'], RL_wt['data']['cell.V'], 'k', label = f'WT, apd90 = {RL_wt["apd"]} ms')
axs[0, 0].legend()
axs[0, 0].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('Membrane potential (mV)')

axs[1, 0].plot(RL_sqt1['data']['Environment.time'], RL_sqt1['data']['cell.V'], 'r', label = f'SQT1, apd90 = {RL_sqt1["apd"]} ms')
axs[1, 0].legend()
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('Current density (pA/pF)')

axs[0, 1].plot(RL_Lcarn_wt['data']['Environment.time'], RL_Lcarn_wt['data']['cell.V'], 'k', ls = 'dotted', label = f'WT + L-Carn, apd90 = {RL_Lcarn_wt["apd"]} ms')
axs[0, 1].legend()
axs[0, 1].set_xlabel('Time (ms)')
axs[0, 1].set_ylabel('Membrane potential (mV)')

axs[1, 1].plot(RL_Lcarn_sqt1['data']['Environment.time'], RL_Lcarn_sqt1['data']['cell.V'], 'r', ls = 'dotted', label = f'SQT1 + L-Carn, apd90 = {RL_Lcarn_sqt1["apd"]} ms')
axs[1, 1].legend()
axs[1, 1].set_xlabel('Time (ms)')
axs[1, 1].set_ylabel('Current density (pA/pF)')

# Tidy up the plots
plt.tight_layout()
plt.show()

# Visualize the results
fig, axs = plt.subplots(2, 1, figsize=(12, 18))

axs[0].plot(RL_wt['data']['Environment.time'], RL_wt['data']['cell.V'], 'k', label = f'WT, apd90 = {RL_wt["apd"]} ms')
axs[0].plot(RL_Lcarn_wt['data']['Environment.time'], RL_Lcarn_wt['data']['cell.V'], 'r', ls = 'dotted', label = f'WT + L-Carn, apd90 = {RL_Lcarn_wt["apd"]} ms')
axs[0].legend()
axs[0].set_title('WT', fontweight = 'bold')
axs[0].set_xlabel('Time (ms)')
axs[0].set_ylabel('Membrane potential (mV)')
axs[0].set_xlim([0, 500])

axs[1].plot(RL_sqt1['data']['Environment.time'], RL_sqt1['data']['cell.V'], 'k', label = f'SQT1, apd90 = {RL_sqt1["apd"]} ms')
axs[1].plot(RL_Lcarn_sqt1['data']['Environment.time'], RL_Lcarn_sqt1['data']['cell.V'], 'r', ls = 'dotted', label = f'SQT1 + L-Carn, apd90 = {RL_Lcarn_sqt1["apd"]} ms')
axs[1].legend()
axs[1].set_title('SQT1', fontweight = 'bold')
axs[1].set_xlabel('Time (ms)')
axs[1].set_ylabel('Membrane potential (mV)')
axs[1].set_xlim([0, 500])

# Tidy up the plots
plt.tight_layout()
plt.show()

#%% IKr, IKs, Ik1 sensitivity analysis
#ik1 wt = 0.81
#ik1 sqt1 = 0.87

# Load the model.
m = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')

# Create an action potential protocol.
p = myokit.Protocol()

# Define the bcl.
bcl = 1000

# Create an event schedule.
p.schedule(1, 20, 0.5, bcl)

def sens_analysis(m, p, x, bcl, prepace, ik1 = True, iks = True, MT = False, carn = False):
    
    # Create a simulation object.
    sim = myokit.Simulation(m, p)
    
    # Set CVODE solver tolerance.
    sim.set_tolerance(1e-8, 1e-8)
    
    # Set the parameters for IKr
    for i in range(len(x)):
        sim.set_constant(f'ikr.p{i+1}', x[i])
    
    # Scale IKs according to conditions
    if iks:
        if MT and carn:
            sim.set_constant('iks.iks_scalar', 0.88)
        elif MT and not carn:
            sim.set_constant('iks.iks_scalar', 1.35)
        elif not MT and carn:
            sim.set_constant('iks.iks_scalar', 0.75)

    # Scale IK1 according to conditions
    if ik1:
        if MT and carn:
            sim.set_constant('ik1.ik1_scalar', 0.87)
        elif not MT and carn:
            sim.set_constant('iks.iks_scalar', 0.81)
    
    # Pre-pace the simulation
    sim.pre(bcl * prepace)
    
    # Run the simulation and calculate the APD90.
    vt = 0.9 * sim.state()[m.get('membrane.V').index()]
    data, apd = sim.run(bcl, log = ['engine.time', 'membrane.V', 'ikr.IKr', 'iks.IKs', 'ik1.IK1'],
                        apd_variable = 'membrane.V', apd_threshold = vt)
    
    # Round the APD90.
    apd90 = round(apd['duration'][0], 2)    
    
    return dict(data = data, apd = apd90)

# WT simulations
wt_sens = sens_analysis(m = m, p = p, x = x_wt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = True, MT = False, carn = False) 

wt_noik1 = sens_analysis(m = m, p = p, x = x_wt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = True, MT = False, carn = False) 

wt_noiks = sens_analysis(m = m, p = p, x = x_wt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = False, MT = False, carn = False) 

wt_ikr = sens_analysis(m = m, p = p, x = x_wt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = False, MT = False, carn = False) 

# WT simulations with L-Carnitine 
wt_carn_sens = sens_analysis(m = m, p = p, x = Lcarn_wt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = True, MT = False, carn = True)  

wt_carn_noik1 = sens_analysis(m = m, p = p, x = Lcarn_wt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = True, MT = False, carn = True) 

wt_carn_noiks = sens_analysis(m = m, p = p, x = Lcarn_wt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = False, MT = False, carn = True) 

wt_carn_ikr = sens_analysis(m = m, p = p, x = Lcarn_wt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = False, MT = False, carn = True) 

# SQT1 simulations
sqt1_sens = sens_analysis(m = m, p = p, x = x_default_sqt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = True, MT = True, carn = False) 

sqt1_noik1 = sens_analysis(m = m, p = p, x = x_default_sqt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = True, MT = True, carn = False)  

sqt1_noiks = sens_analysis(m = m, p = p, x = x_default_sqt, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = False, MT = True, carn = False)  

sqt1_ikr = sens_analysis(m = m, p = p, x = x_default_sqt, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = False, MT = True, carn = False)   

# SQT1 simulations with L-Carnitine
sqt1_carn_sens = sens_analysis(m = m, p = p, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = True, MT = True, carn = True)

sqt1_carn_noik1 = sens_analysis(m = m, p = p, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = True, MT = True, carn = True)  

sqt1_carn_noiks = sens_analysis(m = m, p = p, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, 
                     ik1 = True, iks = False, MT = True, carn = True) 

sqt1_carn_ikr = sens_analysis(m = m, p = p, x = Lcarn_sqt1, bcl = bcl, prepace = 1000, 
                     ik1 = False, iks = False, MT = True, carn = True) 


plt.figure()
plt.plot(wt_sens['data']['engine.time'], wt_sens['data']['membrane.V'], 'k', label = f'All, apd90 = {wt_sens["apd"]} ms')   
plt.plot(wt_noik1['data']['engine.time'], wt_noik1['data']['membrane.V'], 'orange', label = f'No IK1, apd90 = {wt_noik1["apd"]} ms')
plt.plot(wt_noiks['data']['engine.time'], wt_noiks['data']['membrane.V'], 'blue', label = f'No IKs, apd90 = {wt_noiks["apd"]} ms')   
plt.plot(wt_ikr['data']['engine.time'], wt_ikr['data']['membrane.V'], 'purple', label = f'Only IKr, apd90 = {wt_ikr["apd"]} ms')   
plt.legend()    

# Visualize the results
fig, axs = plt.subplots(2, 2, figsize=(12, 18))

axs[0, 0].plot(wt_sens['data']['engine.time'], wt_sens['data']['membrane.V'], 'k', label = f'All, apd90 = {wt_sens["apd"]} ms')   
axs[0, 0].plot(wt_noik1['data']['engine.time'], wt_noik1['data']['membrane.V'], 'orange', label = f'No IK1, apd90 = {wt_noik1["apd"]} ms')
axs[0, 0].plot(wt_noiks['data']['engine.time'], wt_noiks['data']['membrane.V'], 'blue', label = f'No IKs, apd90 = {wt_noiks["apd"]} ms')   
axs[0, 0].plot(wt_ikr['data']['engine.time'], wt_ikr['data']['membrane.V'], 'purple', label = f'Only IKr, apd90 = {wt_ikr["apd"]} ms')   
axs[0, 0].legend()
axs[0, 0].set_title('WT', fontweight = 'bold')
axs[0, 0].set_xlim([0, 500])
axs[0, 0].set_xlabel('Time (ms)')
axs[0, 0].set_ylabel('Membrane potential (mV)')

axs[1, 0].plot(wt_carn_sens['data']['engine.time'], wt_carn_sens['data']['membrane.V'], 'k', label = f'All, apd90 = {wt_carn_sens["apd"]} ms')   
axs[1, 0].plot(wt_carn_noik1['data']['engine.time'], wt_carn_noik1['data']['membrane.V'], 'orange', label = f'No IK1, apd90 = {wt_carn_noik1["apd"]} ms')
axs[1, 0].plot(wt_carn_noiks['data']['engine.time'], wt_carn_noiks['data']['membrane.V'], 'blue', label = f'No IKs, apd90 = {wt_carn_noiks["apd"]} ms')   
axs[1, 0].plot(wt_carn_ikr['data']['engine.time'], wt_carn_ikr['data']['membrane.V'], 'purple', label = f'Only IKr, apd90 = {wt_carn_ikr["apd"]} ms')   
axs[1, 0].legend()
axs[1, 0].set_title('WT + L-Carn', fontweight = 'bold')
axs[1, 0].set_xlim([0, 500])
axs[1, 0].set_xlabel('Time (ms)')
axs[1, 0].set_ylabel('Membrane potential (mV)')

axs[0, 1].plot(sqt1_sens['data']['engine.time'], sqt1_sens['data']['membrane.V'], 'k', label = f'All, apd90 = {sqt1_sens["apd"]} ms')   
axs[0, 1].plot(sqt1_noik1['data']['engine.time'], sqt1_noik1['data']['membrane.V'], 'orange', label = f'No IK1, apd90 = {sqt1_noik1["apd"]} ms')
axs[0, 1].plot(sqt1_noiks['data']['engine.time'], sqt1_noiks['data']['membrane.V'], 'blue', label = f'No IKs, apd90 = {sqt1_noiks["apd"]} ms')   
axs[0, 1].plot(sqt1_ikr['data']['engine.time'], sqt1_ikr['data']['membrane.V'], 'purple', label = f'Only IKr, apd90 = {sqt1_ikr["apd"]} ms')   
axs[0, 1].legend()
axs[0, 1].set_title('SQT1', fontweight = 'bold')
axs[0, 1].set_xlim([0, 500])
axs[0, 1].set_xlabel('Time (ms)')
axs[0, 1].set_ylabel('Membrane potential (mV)')

axs[1, 1].plot(sqt1_carn_sens['data']['engine.time'], sqt1_carn_sens['data']['membrane.V'], 'k', label = f'All, apd90 = {sqt1_carn_sens["apd"]} ms')   
axs[1, 1].plot(sqt1_carn_noik1['data']['engine.time'], sqt1_carn_noik1['data']['membrane.V'], 'orange', label = f'No IK1, apd90 = {sqt1_carn_noik1["apd"]} ms')
axs[1, 1].plot(sqt1_carn_noiks['data']['engine.time'], sqt1_carn_noiks['data']['membrane.V'], 'blue', label = f'No IKs, apd90 = {sqt1_carn_noiks["apd"]} ms')   
axs[1, 1].plot(sqt1_carn_ikr['data']['engine.time'], sqt1_carn_ikr['data']['membrane.V'], 'purple', label = f'Only IKr, apd90 = {sqt1_carn_ikr["apd"]} ms')   
axs[1, 1].legend()
axs[1, 1].set_title('SQT1 + L-Carn', fontweight = 'bold')
axs[1, 1].set_xlim([0, 500])
axs[1, 1].set_xlabel('Time (ms)')
axs[1, 1].set_ylabel('Membrane potential (mV)')


#%% prepace function

def pre_pace(n, t, dur, conduct, carn_list, interval, pp, WT = False, carn = False):
    """
    Pre-pace simulation and data storage.
    
    This function performs pre-pacing simulation using a given model and pacing protocol. The simulation is performed to achieve steady-state before the main pacing simulation. The results from the pre-pacing simulation are saved, and the state data can be used as the initial state for subsequent pacing simulations.
    
    First, cellular pre-pacing is performed, after which 2D tissue pre-pacing is performed. 
    
    Parameters:
    ----------
    n : int
        The number of cells in the 2D simulation grid.
    
    t : int
        The initial pacing time in milliseconds (ms).
    
    dur : int
        The duration of the pre-pacing simulation in milliseconds (ms).
    
    conduct : float
        The cell-cell conductance in the 2D simulation.
    
    carn_list : list of float
        List of parameter values for the iKr (rapid delayed rectifier potassium current) model components when L-carnitine treatment is applied.
    
    interval : float
        The time interval between logged data points during the pre-pacing simulation in milliseconds (ms).
    
    pp : int
        The pre-pacing duration in milliseconds (ms) to reach steady-state.
    
    WT : bool, optional (default = False)
        Flag indicating whether to simulate the pre-pacing for WT (Wild Type) condition.
    
    carn : bool, optional (default = False)
        Flag indicating whether to simulate the pre-pacing with L-carnitine treatment.
    
    Returns:
    -------
    dict
        A dictionary containing the following elements:
        - 'state': The state data obtained after the pre-pacing simulation, which can be used as the initial state for subsequent pacing simulations.
        - 'pre': The logged data during the pre-pacing simulation, including time and membrane potential (V).
        - 'block': The 2D simulation block containing the membrane potential (V) data for all cells in the grid.
    
    Note:
    -----
    - The function uses the 'pre_pace' to simulate pre-pacing and save the results for further analysis.
    - The function uses the 'model' and 'prot' objects, which should be set up before calling this function.
    - The 'conduct' parameter represents cell-cell conductance in the 2D simulation, which influences the electrical coupling between cells.
    - The 'carn_list' parameter contains the parameter values for the iKr model when L-carnitine treatment is applied.
    - The 'interval' parameter specifies the time interval between logged data points during the pre-pacing simulation.
    - Use 'WT=True' to simulate pre-pacing for the Wild Type condition.
    - Use 'carn=True' to simulate pre-pacing with L-carnitine treatment.
    - The results from the pre-pacing simulation can be saved in numpy files for further analysis and visualization.
    - The function returns a dictionary containing the logged data, state data, and 2D simulation block for analysis and visualization.
    """

    
    # Load the model and set the input parameters.
    model = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
    model.set_value('cell.mode', 0)
    
    # Initialize the default parameters
    default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
    default_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1]
    
    # Initialize a protocol.
    prot = myokit.Protocol()
    
    # Set the bcl to 1000 ms.
    bcl = 1000
    
    hz = 1/(bcl/1000)
    
    # create an event schedule.
    prot.schedule(1, 20, 0.5, bcl, 0 )
  
    # Create a simulation object.
    sim_pre = myokit.Simulation(model, prot)
    
    # Pre-pace the model to steady state.
    sim_pre.pre(pp)
    
    # Save the pre-paced states.
    pre_state = sim_pre.state() 
        
    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)
    s = myokit.SimulationOpenCL(model, p, ncells = [n, n])
    s.set_paced_cells(3, n, 0, 0)
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    s.set_state(pre_state) 
    
    # Conditional flags.
    if WT is False: 
        for i in range(len(default_sqt)):
            s.set_constant(f'ikr.p{i+1}', default_sqt[i])
        s.set_constant('iks.iks_scalar', 1.35)
    else:
        for i in range(len(default_wt)):
            s.set_constant(f'ikr.p{i+1}', default_wt[i])
        s.set_constant('iks.iks_scalar', 1)
    
    if carn is True:
        for i in range(len(carn_list)):
            s.set_constant(f'ikr.p{i+1}', carn_list[i])
        s.set_constant('iks.iks_scalar', 0.88)
  
    # Generate the first depolarization.
    pre = s.run(dur, log = ['engine.time', 'membrane.V'], log_interval = interval)
    state = s.state()
    block = pre.block2d()
    
    # Save the list with nympy.
    if WT is True:
        if carn is True: 
            np.save(f'pre_pace{pp}_WT_carn_{dur}_cell_{n}_iks_{hz}Hz.npy', state)
            block.save(f'2D_sim_WT_carn_pre{pp}_cell_{n}_iks_{hz}Hz.zip')
        else: 
            np.save(f'pre_pace_{pp}WT_{dur}_cell_{n}_iks_{hz}Hz.npy', state)  
            block.save(f"2D_sim_WT_pre{pp}_cell_{n}_iks_{hz}Hz.zip")  
    else:
        if carn is True: 
            np.save(f'pre_pace{pp}_MT_carn_{dur}_cell_{n}_iks_{hz}Hz.npy', state)
            block.save(f"2D_sim_MT_carn_pre{pp}_cell_{n}_iks_{hz}Hz.zip")
        else: 
            np.save(f'pre_pace{pp}_MT_new{dur}_cell_{n}_iks_{hz}Hz.npy', state)
            block.save(f"2D_sim_MT_newpre{pp}_cell_{n}_iks_{hz}Hz.zip")
      
    return(dict(state = state, pre = pre, block = block))

# Run the 2D pre-pace function. 
WT_pre_pace = pre_pace(n = 400, t = 1000, dur = 10000, conduct = 9, interval = 5, carn_list = Lcarn_sqt1, pp = 2000000, WT = True, carn = False)
MT_pre_pace = pre_pace(n = 400, t = 1000, dur = 10000, conduct = 9, interval = 5, carn_list = Lcarn_sqt1, pp = 2000000, WT = False, carn = False)
MT_carn_pre_pace = pre_pace(n = 400, t = 1000, dur = 10000, conduct = 9, interval = 5, carn_list = Lcarn_sqt1, pp = 2000000, WT = False, carn = True)
#%%  MT reentry function.

def vulnerability_window_MT(inputs):
    """Calculate vulnerability window for specific input(s).

    This function calculates the vulnerability window for a specific S1S2 input value (or a range of S1S2) by simulating
    electrical activity in a cardiac tissue model. The vulnerability window is the time interval
    during which the tissue is susceptible to re-entry circuits.

    Parameters:
    inputs (list): The S1S2 interval.

    Returns:
    dict: A dictionary containing simulation logs, simulation blocks, and re-entry time.

    Simulation Logs:
    The function returns the simulation logs, which contain time and membrane voltage data.

    Simulation Blocks:
    The simulation logs are converted to a 2D simulation block, representing the spatiotemporal
    distribution of the membrane voltage during the simulation.

    Re-entry Time:
    The function calculates and returns the re-entry time, which is the time interval during which
    the tissue exhibits re-entry circuits. 

    Note:
    - The function uses an OpenCL-based cardiac tissue model with specific input parameters.
    - The function saves the simulation block and re-entry time for further analysis.
    - The model parameters and pacing protocol can be modified based on the specific requirements.
    - The function supports options to simulate with different model variants (WT or mutant) and
      with or without the effect of L-carnitine (carn).

    Example:
    MT_S1S2 = vulnerability_window_MT([200, 210, 220])

    """
    
    # Set default parameters.
    n = 400
    t = 1000
    dur = 100
    dur_sim = 1000
    conduct = 9
    interval = 5
    carn_list = Lcarn_sqt1
    pp = 2000000
    WT = False
    carn = False
    
    # Initialize the model.
    model = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
    model.set_value('cell.mode', 0)
    
    # Set the default IKr parameters for WT and SQT1.
    default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
    default_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1 ]
    
    # Determine the Hz.
    hz = 1/(t/1000)
    
    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)
    s = myokit.SimulationOpenCL(model, p, ncells=[n, n])
    s.set_paced_cells(3, n, 0, 0)
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    
    if not WT: 
        param_list = default_sqt
    else:
        param_list = default_wt
    
    if carn:
        param_list = carn_list
    
    for i in range(len(param_list)):
        s.set_constant(f'ikr.p{i+1}', param_list[i])
    
    # Set IKs.
    s.set_constant('iks.iks_scalar', 1.35)
    
    # Load the pre-pacing state.
    pre_pace = np.load('pre_pace2000000_MT_new10000_cell_400_iks_1.0Hz.npy')
    s.set_state(pre_pace)
    
    # Run the model for the S1.
    log = s.run(dur, log=['engine.time', 'membrane.V'], log_interval=interval)
    
    # Perform the simulation for 10 seconds
    for i in range(10):
        p2 = myokit.pacing.blocktrain(t, 1, inputs, 1, 1)
        s.set_protocol(p2)
        s.set_paced_cells(n/2, n/2, 0, 0)
    
        log = s.run(dur_sim, log=log, log_interval=interval)
        block = log.block2d()
        
        # If no more electrical activity is detected (i.e., no re-entry), then stop the simulation.
        vm = block.get2d('membrane.V')
        maxval = np.max(vm[-1].flatten())
        if maxval < -50:
            print('no more activity detected')
            break
        
        # Save the results as .zips to be loaded in the block viewer.
        if WT:
            if carn: 
                block.save(f'2D_sim_WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip')
            else: 
                block.save(f"2D_sim_WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")    
        else:
            if carn: 
                block.save(f"2D_sim_MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
            else: 
                block.save(f"2D_sim_MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
    
    # Calculate the reentry time based on a threshold of -80 mV.
    reentry_time = calculate_reentry_time(i=i, vm=vm, dur_sim=dur_sim, dur=dur, s2=inputs, interval=interval, cutoff=-80)  
    if WT:
        if carn: 
            np.save(f'WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy', reentry_time)
        else: 
            np.save(f"WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)    
    else:
        if carn: 
            np.save(f"MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)
        else: 
            np.save(f"MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)

    return dict(log=log, block=block, time=reentry_time)


#%% WT reentry function
def vulnerability_window_WT(inputs):
    """Calculate vulnerability window for specific input(s).

    This function calculates the vulnerability window for a specific S1S2 input value (or a range of S1S2) by simulating
    electrical activity in a cardiac tissue model. The vulnerability window is the time interval
    during which the tissue is susceptible to re-entry circuits.

    Parameters:
    inputs (list): The S1S2 interval.

    Returns:
    dict: A dictionary containing simulation logs, simulation blocks, and re-entry time.

    Simulation Logs:
    The function returns the simulation logs, which contain time and membrane voltage data.

    Simulation Blocks:
    The simulation logs are converted to a 2D simulation block, representing the spatiotemporal
    distribution of the membrane voltage during the simulation.

    Re-entry Time:
    The function calculates and returns the re-entry time, which is the time interval during which
    the tissue exhibits re-entry circuits. 

    Note:
    - The function uses an OpenCL-based cardiac tissue model with specific input parameters.
    - The function saves the simulation block and re-entry time for further analysis.
    - The model parameters and pacing protocol can be modified based on the specific requirements.
    - The function supports options to simulate with different model variants (WT or mutant) and
      with or without the effect of L-carnitine (carn).

    Example:
    MT_S1S2 = vulnerability_window_MT([200, 210, 220])

    """
    
    # Set default parameters.
    n = 400
    t = 1000
    dur = 100
    dur_sim = 1000
    conduct = 9
    interval = 5
    carn_list = Lcarn_sqt1
    pp = 2000000
    WT = True
    carn = False
    
    # Initialize the model.
    model = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
    model.set_value('cell.mode', 0)
    
    # Set the default IKr parameters for WT and SQT1.
    default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
    default_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1 ]
    
    # Determine the Hz.
    hz = 1/(t/1000)
    
    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)
    s = myokit.SimulationOpenCL(model, p, ncells=[n, n])
    s.set_paced_cells(3, n, 0, 0)
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    
    if not WT: 
        param_list = default_sqt
    else:
        param_list = default_wt
    
    if carn:
        param_list = carn_list
    
    for i in range(len(param_list)):
        s.set_constant(f'ikr.p{i+1}', param_list[i])
    
    # Set IKs.
    s.set_constant('iks.iks_scalar', 1.35)
    
    # Load the pre-pacing state.
    pre_pace = np.load('pre_pace2000000_MT_new10000_cell_400_iks_1.0Hz.npy')
    s.set_state(pre_pace)
    
    # Run the model for the S1.
    log = s.run(dur, log=['engine.time', 'membrane.V'], log_interval=interval)
    
    # Perform the simulation for 10 seconds
    for i in range(10):
        p2 = myokit.pacing.blocktrain(t, 1, inputs, 1, 1)
        s.set_protocol(p2)
        s.set_paced_cells(n/2, n/2, 0, 0)
    
        log = s.run(dur_sim, log=log, log_interval=interval)
        block = log.block2d()
        
        # If no more electrical activity is detected (i.e., no re-entry), then stop the simulation.
        vm = block.get2d('membrane.V')
        maxval = np.max(vm[-1].flatten())
        if maxval < -50:
            print('no more activity detected')
            break
        
        # Save the results as .zips to be loaded in the block viewer.
        if WT:
            if carn: 
                block.save(f'2D_sim_WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip')
            else: 
                block.save(f"2D_sim_WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")    
        else:
            if carn: 
                block.save(f"2D_sim_MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
            else: 
                block.save(f"2D_sim_MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
    
    # Calculate the reentry time based on a threshold of -80 mV.
    reentry_time = calculate_reentry_time(i=i, vm=vm, dur_sim=dur_sim, dur=dur, s2=inputs, interval=interval, cutoff=-80)  
    if WT:
        if carn: 
            np.save(f'WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy', reentry_time)
        else: 
            np.save(f"WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)    
    else:
        if carn: 
            np.save(f"MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)
        else: 
            np.save(f"MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)

    return dict(log=log, block=block, time=reentry_time)

#%% MT + L-carn reentry
Lcarn_sqt1 = [0.017787,-48.520307,14.325950,22.303676,6.877993,0.000241,14.842432,-5.368071,-3.843856,4.941128,2.061902]

def vulnerability_window_MT_Carn(inputs):
    """Calculate vulnerability window for specific input(s).

    This function calculates the vulnerability window for a specific S1S2 input value (or a range of S1S2) by simulating
    electrical activity in a cardiac tissue model. The vulnerability window is the time interval
    during which the tissue is susceptible to re-entry circuits.

    Parameters:
    inputs (list): The S1S2 interval.

    Returns:
    dict: A dictionary containing simulation logs, simulation blocks, and re-entry time.

    Simulation Logs:
    The function returns the simulation logs, which contain time and membrane voltage data.

    Simulation Blocks:
    The simulation logs are converted to a 2D simulation block, representing the spatiotemporal
    distribution of the membrane voltage during the simulation.

    Re-entry Time:
    The function calculates and returns the re-entry time, which is the time interval during which
    the tissue exhibits re-entry circuits. 

    Note:
    - The function uses an OpenCL-based cardiac tissue model with specific input parameters.
    - The function saves the simulation block and re-entry time for further analysis.
    - The model parameters and pacing protocol can be modified based on the specific requirements.
    - The function supports options to simulate with different model variants (WT or mutant) and
      with or without the effect of L-carnitine (carn).

    Example:
    MT_S1S2 = vulnerability_window_MT([200, 210, 220])
    """
    
    # Set default parameters.
    n = 400
    t = 1000
    dur = 100
    dur_sim = 1000
    conduct = 9
    interval = 5
    carn_list = Lcarn_sqt1
    pp = 2000000
    WT = False
    carn = True
    
    # Initialize the model.
    model = myokit.load_model('MMT/ORD_LOEWE_CL_adapt.mmt')
    model.set_value('cell.mode', 0)
    
    # Set the default IKr parameters for WT and SQT1.
    default_sqt = [0.029412, -38.65, 19.46, 16.49, 6.76, 0.0003, 14.1, -5, -3.3328, 5.1237, 2]
    default_wt = [0.029412, 15, 22.4, 14.1, 6.5, 0.0003, 14.1, -5, -3.3328, 5.1237, 1 ]
    
    # Determine the Hz.
    hz = 1/(t/1000)
    
    # Create a pacing protocol.
    p = myokit.pacing.blocktrain(t, 1, 0, 1, 0)
    s = myokit.SimulationOpenCL(model, p, ncells=[n, n])
    s.set_paced_cells(3, n, 0, 0)
    s.set_conductance(conduct, conduct)
    s.set_step_size(0.01)
    
    if not WT: 
        param_list = default_sqt
    else:
        param_list = default_wt
    
    if carn:
        param_list = carn_list
    
    for i in range(len(param_list)):
        s.set_constant(f'ikr.p{i+1}', param_list[i])
    
    # Set IKs.
    s.set_constant('iks.iks_scalar', 1.35)
    
    # Load the pre-pacing state.
    pre_pace = np.load('pre_pace2000000_MT_new10000_cell_400_iks_1.0Hz.npy')
    s.set_state(pre_pace)
    
    # Run the model for the S1.
    log = s.run(dur, log=['engine.time', 'membrane.V'], log_interval=interval)
    
    # Perform the simulation for 10 seconds
    for i in range(10):
        p2 = myokit.pacing.blocktrain(t, 1, inputs, 1, 1)
        s.set_protocol(p2)
        s.set_paced_cells(n/2, n/2, 0, 0)
    
        log = s.run(dur_sim, log=log, log_interval=interval)
        block = log.block2d()
        
        # If no more electrical activity is detected (i.e., no re-entry), then stop the simulation.
        vm = block.get2d('membrane.V')
        maxval = np.max(vm[-1].flatten())
        if maxval < -50:
            print('no more activity detected')
            break
        
        # Save the results as .zips to be loaded in the block viewer.
        if WT:
            if carn: 
                block.save(f'2D_sim_WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip')
            else: 
                block.save(f"2D_sim_WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")    
        else:
            if carn: 
                block.save(f"2D_sim_MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
            else: 
                block.save(f"2D_sim_MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_iks_{hz}Hz.zip")
    
    # Calculate the reentry time based on a threshold of -80 mV.
    reentry_time = calculate_reentry_time(i=i, vm=vm, dur_sim=dur_sim, dur=dur, s2=inputs, interval=interval, cutoff=-80)  
    if WT:
        if carn: 
            np.save(f'WT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy', reentry_time)
        else: 
            np.save(f"WT_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)    
    else:
        if carn: 
            np.save(f"MT_carn_cell{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)
        else: 
            np.save(f"MT_cell_{inputs}_{n}_cond{conduct}_PP{pp}_reentrytime_iks_{hz}Hz.npy", reentry_time)

    return dict(log=log, block=block, time=reentry_time)

#%% Parallelization of the S1S2 simulations to speed it up.

if __name__ == '__main__':
    # Record the start time of the script execution.
    start_time = time.time()
    
    # Initialize an empty list to store the final results.
    final_results = []
    
    # Create a list of S1S2 values to iterate over.
    my_list = list(range(200, 210, 10))
    
    # Set the value of 'cellular pre-pacing' to 2000000 ms.
    pp = 2000000
    
    # Determine the number of CPU cores available and create a Pool object with a maximum number of processes.
    # This pool of processes will be used to perform parallel computations on the elements of 'my_list'.
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=min(num_processes, len(my_list)))
    
    # Apply the function 'vulnerability_window_MT' to each element of 'my_list' in parallel.
    # The 'imap' method returns an iterator that yields the results of the function calls in the order of input.
    results = pool.imap(vulnerability_window_MT, my_list)
    
    # Collect the results obtained from each iteration and store them in the 'final_results' list.
    # Also, save the simulation blocks obtained from each result to a file with a specific naming convention.
    for i, result in enumerate(results):
        final_results.append(result)
        result['block'].save(f'2D_sim_MT_cell_res{my_list[i]}_400_conduct9_PP{pp}_iks_1Hz.zip')

    # Close the Pool to prevent any more tasks from being submitted to it.
    pool.close()
    
    # Wait for all the worker processes to finish and terminate the Pool.
    pool.join()
    
    # Record the end time of the script execution.
    end_time = time.time()
    
    # Calculate the total time taken for the script execution.
    total_time_taken = end_time - start_time
    
    # If no error occurred during script execution, print the total time taken in seconds.
    print(f"Time taken: {total_time_taken} seconds")

    
#%% Import all the reentry durations.

def reentry_df(reentry_range, WT = True, Carn = False, save = False):

    """
    Creates a pandas DataFrame from a range of reentry values and specified conditions.

    Args:
        reentry_range (List[int]): A list of reentry values.
        WT (bool, optional): Indicates whether WT condition is True or False. Defaults to True.
        Carn (bool, optional): Indicates whether Carn condition is True or False. Defaults to False.

    Returns:
        pd.DataFrame: A DataFrame containing the reentry values.

    Raises:
        FileNotFoundError: If the file corresponding to the given conditions and reentry value is not found.

    """
    
    # Initialize some conditions. 
    npy_list = []
    file_format = '{}{}_400_cond9_PP2000000_reentrytime_iks.npy'
    file_prefix = 'WT' if WT else 'MT'
    file_suffix = '_carn' if Carn else ''
 
    # Loop over the reentry values and try to load corresponding files
    for i in reentry_range:
        if WT is False:
            if Carn is False:
                file_name = file_format.format(file_prefix + file_suffix + '_cell_', i)
            else:
                file_name = file_format.format(file_prefix + file_suffix + '_cell', i)
        else:
            file_name = file_format.format(file_prefix + file_suffix + '_cell', i)
        
        try:
            # Load the reentry duration data from the file and add it to the list
            reentry = np.load(file_name, allow_pickle=True)
            npy_list.append((i, reentry))
        except FileNotFoundError as e:
            # If the file is not found, raise an error with an informative message
            raise FileNotFoundError(f"File '{file_name}' not found.") from e

    # Create a DataFrame from the list of reentry values
    df = pd.DataFrame(npy_list, columns=['Number', 'Data']).astype(int)
    
    # If the 'save' flag is True, save the DataFrame to a CSV file
    if save:
        save_file_name = f"{file_prefix}{file_suffix}_reentrydur_1Hz.csv"
        df.to_csv(save_file_name, index=False)
    
    return df


# Define the S1S2 ranges for each condition.
WT_range = np.arange(240, 410, 10) 
MT_range = np.arange(200, 410, 10)         
MT_Carn_range = np.arange(200, 410, 10)      

# Calculate the reentry durations for each of the S1S2 intervals.
WT_reentry = reentry_df(WT_range, WT = True, Carn = False, save = True)       
MT_reentry = reentry_df(MT_range, WT = False, Carn = False, save = True)
MT_Carn_reentry = reentry_df(MT_Carn_range, WT = False, Carn = True, save = True)

# Alternatively, you can also load the data from the folder. 
WT_reentrydur = pd.read_csv('Data/WT_reentrydur_1Hz.csv')
MT_reentrydur = pd.read_csv('Data/MT_reentrydur_1Hz.csv')
MT_Carn_reentrydur = pd.read_csv('Data/MT_carn_reentrydur_1Hz.csv')

# Calculate the total arrhythmogenicity (in s) by summing the reentries.
WT_reentrytotal = WT_reentrydur['Data'].sum()/1000
MT_reentrytotal = MT_reentrydur['Data'].sum()/1000
MT_Carn_reentrytotal = MT_Carn_reentrydur['Data'].sum()/1000


