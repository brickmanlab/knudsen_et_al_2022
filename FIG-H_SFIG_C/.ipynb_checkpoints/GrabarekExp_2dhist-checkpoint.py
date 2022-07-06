#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import collections
import operator
from collections import OrderedDict

from random import random
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
import random

from System_004_definer import *
from plotting_rules import *
from Gillespie_004 import Gillespie_run_v1 as Gillespie_run
import pickle

import seaborn as sns




def sample_trajectories(initial_state, network_ID, N_samples = 10, tmax = 300):
    states_dict = []
    for i in range(N_samples):
        System = System_simplified(initial_state, network_ID = network_ID)
        GillespieSystem = Gillespie_run(System, initial_state, timestop = tmax)
        states_dict.append({'t':np.array(GillespieSystem.time), 'ID':i, 'G':np.array(GillespieSystem.G), 'N':np.array(GillespieSystem.N)})
    return states_dict

def find_states_in_timewindow_2D(states_dict,t, dt=1):
    N_samples = np.shape(states_dict)[0]
    values = []
    for state in states_dict:
        idx =  np.where(np.logical_and(state['t']>=(t-dt/2), state['t']<=(t+dt/2)))
        if np.shape(idx)[1]:
            values.append([state['G'][idx[0][0]], state['N'][idx[0][0]]])
        else:
            print('did not find values at t=', t)
    return np.array(values)


def calculate_probability(states, threshold = 5.5):
    N_tot = len(states)
    idx = np.where(states >= threshold)
    n_high = len(idx[0])
    return n_high/N_tot

def sample_probability_vs_time(initial_state, state_thresholds, network_ID, N_samples):
    samp = sample_trajectories(initial_state, network_ID, N_samples)
    
    timepoints = np.linspace(0,200, 200)
    states_at_t = []
    pNG_at_t = []
    for t in timepoints:
        ng_states = find_states_in_timewindow_2D(samp,t=t, dt=1)
        states_at_t.append(ng_states)
        p_G = calculate_probability(ng_states[:,0], state_thresholds[0])
        p_N = calculate_probability(ng_states[:,1], state_thresholds[1])

        #print(np.shape(np.array([p_G, p_N]).reshape(1,2)))
        #pNG_at_t.append(np.array([p_G, p_N]).reshape(2,1))
        pNG_at_t.append([p_G, p_N])
        #print(np.shape(pNG_at_t))
    return timepoints, np.array(pNG_at_t)


# In[171]:


def sample_net2_net3_probab_diff(initial_state, fname, state_tresholds, N_samples):
    fname_res = fname[:-4] + '_N' + str(N_samples) + '.pickle'
    try:
        with open(fname_res, 'rb') as handle:
            print('Loading form file', fname_res)
            res = pickle.load(handle)

    except:
        print('Coudlnt open file', fname_res)
        t, pNG_2 = sample_probability_vs_time(initial_state, statte_thresholds, network_ID=2, N_samples=N_samples)
        t, pNG_3 = sample_probability_vs_time(initial_state, statte_thresholds, network_ID=3, N_samples=N_samples)
        pNG_diff = pNG_3 - pNG_2

        res = {}

        ic = [initial_state['G'][0], initial_state['N'][0], initial_state['E'][0]]
        res['ic'] = ic
        res['t'] = t
        res['pNG_2'] = pNG_2
        res['pNG_3'] = pNG_3
        res['pNG_diff'] = pNG_diff
        with open(fname_res, 'wb') as handle:
            pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return res

#res = sample_net2_net3_probab_diff(initial_state, fname, N_samples=10)

# add argument ics_id, to put into the file name
def sample_net2_net3_probab_diff_v2(ics, ics_id, statte_thresholds, fname, N_samples, n_ics = 1000):
    fname_res = fname[:-4] +'icid_' + str(ics_id) + '_N' + str(N_samples) + '.pickle'
    try:
        with open(fname_res, 'rb') as handle:
            print('Loading form file', fname_res)
            results = pickle.load(handle)

    except:
        print('Coudlnt open file', fname_res)
        results = []
        count = 0
        for ic in ics:
            if count>n_ics:
                break
            else:
                initial_state = {}
                count+=1
                if count%5==1:
                    print('Countdown of # of ics', count)
                initial_state['G'] = [ic[0]]
                initial_state['N'] = [ic[1]]
                initial_state['E'] = [ic[2]]

                t, pNG_2 = sample_probability_vs_time(initial_state,statte_thresholds, network_ID=2, N_samples=N_samples)
                t, pNG_3 = sample_probability_vs_time(initial_state,statte_thresholds, network_ID=3, N_samples=N_samples)
                pNG_diff = pNG_3 - pNG_2

                res = {}

                ic = [initial_state['G'][0], initial_state['N'][0], initial_state['E'][0]]
                res['ic'] = ic
                res['t'] = t
                res['pNG_2'] = pNG_2
                res['pNG_3'] = pNG_3
                res['pNG_diff'] = pNG_diff
                results.append(res)
        with open(fname_res, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return results




def sample_net2_net3_probab_diff_v3(ics_2, ics_3, ics_id, statte_thresholds, fname, N_samples, n_ics = 1000):
    ''' Here we use ic_2 for net2 and ic_3 for net3. The ic_2 and ic_3 are related, i.e. run form the same 0-ic but correspondingly 2-net and 3-net. have to use the same row number.'''
    fname_res = fname[:-4] +'_icid_' + str(ics_id) + '_N' + str(N_samples) + '_v3.pickle'
    try:
        with open(fname_res, 'rb') as handle:
            print('Loading form file', fname_res)
            results = pickle.load(handle)

    except:
        print('Coudlnt open file', fname_res)
        results = []
        count = 0
        for ic2,ic3 in zip(ics_2,ics_3):
            if count>n_ics:
                break
            else:
                count+=1
                if count%5==1:
                    print(count)
               
                res = {}

                initial_state = {}
                initial_state['G'] = [ic2[0]]
                initial_state['N'] = [ic2[1]]
                initial_state['E'] = [ic2[2]]
                res['ic2'] = ic2
               
                t, pNG_2 = sample_probability_vs_time(initial_state,statte_thresholds, network_ID=2, N_samples=N_samples)
                  
                initial_state = {}
                initial_state['G'] = [ic3[0]]
                initial_state['N'] = [ic3[1]]
                initial_state['E'] = [ic3[2]]
                res['ic3'] = ic3
                t, pNG_3 = sample_probability_vs_time(initial_state,statte_thresholds, network_ID=3, N_samples=N_samples)
                
                pNG_diff = pNG_3 - pNG_2

               
                #ic = [initial_state['G'][0], initial_state['N'][0], initial_state['E'][0]]
                res['t'] = t
                res['pNG_2'] = pNG_2
                res['pNG_3'] = pNG_3
                res['pNG_diff'] = pNG_diff
                results.append(res)
        with open(fname_res, 'wb') as handle:
            pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


    return results
def pool_results_across_ic(results):
    pG_diff = []
    pN_diff = []
    time = []
    for res in results:
        pG_diff.append(res['pNG_diff'][:,0])
        pN_diff.append(res['pNG_diff'][:,1])
        time = res['t']
        #print(time)
    return pG_diff, pN_diff, time

# 1) make dataframe with coloumns named by time and rows condaint prob differneces

def convert_results_to_df(times, pG_diff):
    keys = [f"t={t:.2f}" for t in times]
    values = [ np.array(pG_diff)[:,idx_t] for idx_t in range(len(times))]
    df = pd.DataFrame(dict(zip(keys, values)))
    return df





import pandas as pd
from sklearn.neighbors import KernelDensity

import matplotlib as mpl
import matplotlib.gridspec as grid_spec

#times = [x for x in np.unique(data.country)]
def ridge_plot_probab_diff(df,title):
    
    
    n_times = 4#len(df.iloc[0])
    times = list(df.columns)
    gs = (grid_spec.GridSpec(n_times,1))

    fig = plt.figure(figsize=(8,6))
    plt.title(title)
    #creating empty list
    ax_objs = []

    for i in range(1,n_times):
        # creating new axes object and appending to ax_objs
            ax_objs.append(fig.add_subplot(gs[i:i+1, 0:]))
        # plotting the distribution
            s = pd.Series(df.iloc[:,i])
            s_last = pd.Series(df.iloc[:,-1])
            print(df.iloc[:,-50:].mean().mean())
            s_last_50 = pd.Series(df.iloc[:,-50:].mean().mean())
            
            #print(np.shape(df.iloc[:,-1]))
            #print(df.iloc[:,-1])
            #print(s_last)
            fr_pos_s = len(s[s > 0])/len(s)
            fr_pos_s_last = len(s_last[s_last > 0])/len(s_last)
            avg_s = s.mean()
            avg_s_last = s_last.mean()
            avg_s_last_50 = s_last_50.mean()
            print(avg_s_last_50)
            
            dff = pd.DataFrame({
                'x': s,
                'y': s_last,
                })
            
            
            plot = dff.plot.kde(bw_method=0.3,ax=ax_objs[-1],color="grey", lw=1)
            # grabbing x and y data from the kde plot
            x = plot.get_children()[0]._x
            y = plot.get_children()[0]._y

            # filling the space beneath the distribution
            ax_objs[-1].fill_between(x,y,color='purple')

            # setting uniform x and y lims
            ax_objs[-1].set_xlim(-1, 1)
            ax_objs[-1].set_ylim(0,10)
            # make background transparent
            rect = ax_objs[-1].patch
            rect.set_alpha(0)

            # remove borders, axis ticks, and labels
           # ax_objs[-1].set_yticklabels([])
            #ax_objs[-1].set_ylabel('')
            title = str(times[i]) + f' bias = {fr_pos_s:.2f}  ref_bias = {fr_pos_s_last:.2f}'
            title = str(times[i]) + f' mean = {avg_s:.2f}  ref_mean = {avg_s_last:.2f} ref_mean_last50 = {avg_s_last_50:.2f}'
            ax_objs[-1].set_title(title)
            if i == n_times-1:
                ax_objs[-1].set_xlabel('P_diff')
                #pass
            else:
                ax_objs[-1].set_xticklabels([])
                
            spines = ["top","right","left","bottom"]
            for s in spines:
                ax_objs[-1].spines[s].set_visible(False)

    plt.tight_layout()
    plt.show()
    
def plot_p2_p3_hist(df_2,df_3, title):
    n_times = 4#len(df.iloc[0])
    times = list(df_2.columns)
    h = []
    for i in range(1,n_times):
        # creating new axes object and appending to ax_objs
        plt.figure()
        # plotting the distribution
        s_2 = pd.Series(df_2.iloc[:,i])
        s_3 = pd.Series(df_3.iloc[:,i])
        plt.hist2d(s_2, s_3, bins=(60, 60), cmap=plt.cm.Greys, range = [[0,0.6], [0,0.6]], density=True, vmin=0, vmax=20)
        plt.xlim(0,0.6)
        plt.ylim(0,0.6)
        plt.colorbar()
        #plt.savefig()
        #plt.show()
        
        hh,x_edges, y_edges,_ = plt.hist2d(s_2, s_3, bins=(60, 60), cmap=plt.cm.jet, range = [[0,0.6], [0,0.6]], density=True, vmin=0, vmax=20)
        h.append(hh)
    return h, x_edges, y_edges

def pool_results_across_ic(results):
    pG_diff = []
    pN_diff = []
    pNG_diff = []
    pNG_2 = []
    pNG_3 = []
    
    time = []
    for res in results:
        pG_diff.append(res['pNG_diff'][:,0])
        pN_diff.append(res['pNG_diff'][:,1])
        pNG_diff.append(res['pNG_diff'])
        pNG_2.append(res['pNG_2'])
        pNG_3.append(res['pNG_3'])
        
        
        #pG_rel_diff.append(res['pNG_rel_diff'][:,0])
        #pN_rel_diff.append(res['pNG_rel_diff'][:,1])
        
        time = res['t']
        #print(time)
    return pG_diff, pN_diff, pNG_diff, pNG_2, pNG_3, time


    
def test_p_diff(ics, ics_id, state_thresholds, fname, N_samples=20, pG = True, relative = False, p2_p3_switch = True):
    results = sample_net2_net3_probab_diff_v2(ics, ics_id, state_thresholds, fname, N_samples)   # here we take the same IC, i.e. for net2
    
    pG_diff,pN_diff,pNG_diff, pNG_2, pNG_3, t = pool_results_across_ic(results)
    
        
    # make dataframe with columns named by time and each row with prob differneces for specific IC
    np.seterr(divide='ignore', invalid='ignore')
      
    if pG and not relative:#choose if want to show pG or pN_diff, i.e. switchin to G state or to N state
        df = convert_results_to_df(t,pG_diff)
    elif not pG and not relative:
        df = convert_results_to_df(t,pN_diff)
    elif pG and relative:
        pG_rel_diff = (np.array(pNG_diff)/np.array(pNG_2))[:,:,0]
        df = convert_results_to_df(t,pG_rel_diff)
    else:
        pN_rel_diff = (np.array(pNG_diff)/np.array(pNG_2))[:,:,1]
        df = convert_results_to_df(t,pN_rel_diff)
    #display((df.iloc[:,1]))
    #display((df.iloc[0]))
    
    title = fname[:-4] + ' N=' + str(N_samples) + ' ics_id=' + str(ics_id) 
    #ridge_plot_probab_diff(df,title=title )

    if p2_p3_switch:
        if pG:
            df_2 = convert_results_to_df(t,pNG_2[:,:,0])
            df_3 = convert_results_to_df(t,pNG_3[:,:,0])     
        else:
            df_2 = convert_results_to_df(t,np.array(pNG_2)[:,:,1])
            df_3 = convert_results_to_df(t,np.array(pNG_3)[:,:,1])     
        #plot_p2_p3_hist(df_2,df_3, title=title)
        hist, x_edges, y_edges = plot_p2_p3_hist(df_2,df_3, title=title)
        return hist, x_edges, y_edges


def test_p_diff_v3(ics2, ics3, ics_id, state_thresholds, fname, N_samples=20, pG = True, relative = False, p2_p3_switch = True):
    #results = sample_net2_net3_probab_diff_v2(ics, ics_id, state_thresholds, fname, N_samples)   # here we take the same IC, i.e. for net2
    results = sample_net2_net3_probab_diff_v3(ics2,ics3, ics_id, state_thresholds, fname, N_samples)   

    pG_diff,pN_diff,pNG_diff, pNG_2, pNG_3, t = pool_results_across_ic(results)
    
    np.seterr(divide='ignore', invalid='ignore')
    # make dataframe with columns named by time and each rwo with prob differneces for specific IC
    if pG and not relative:#choose if want to show pG or pN_diff, i.e. switchin to G state or to N state
        df = convert_results_to_df(t,pG_diff)
    elif not pG and not relative:
        df = convert_results_to_df(t,pN_diff)
    elif pG and relative:
        pG_rel_diff = (np.array(pNG_diff)/np.array(pNG_2))[:,:,0]
        df = convert_results_to_df(t,pG_rel_diff)
    else:
        pN_rel_diff = (np.array(pNG_diff)/np.array(pNG_2))[:,:,1]
        df = convert_results_to_df(t,pN_rel_diff)
    #display((df.iloc[:,1]))
    #display((df.iloc[0]))
    title = fname[:-4] + ' N=' + str(N_samples) + ' ics_id=' + str(ics_id) 
    #ridge_plot_probab_diff(df,title=title )
    
    if p2_p3_switch:
        if pG:
            df_2 = convert_results_to_df(t,np.array(pNG_2)[:,:,0])
            df_3 = convert_results_to_df(t,np.array(pNG_3)[:,:,0])     
        else:
            df_2 = convert_results_to_df(t,np.array(pNG_2)[:,:,1])
            df_3 = convert_results_to_df(t,np.array(pNG_3)[:,:,1])     
        #plot_p2_p3_hist(df_2,df_3, title=title)
        hist, x_edges, y_edges = plot_p2_p3_hist(df_2,df_3, title=title)
        return hist, x_edges, y_edges
    