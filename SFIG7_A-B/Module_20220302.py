#!/usr/bin/env python
# coding: utf-8

# In[1]:


from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from numpy import random
from matplotlib.collections import PathCollection

from scipy.stats import loguniform



# # Equations and parameters

# In[2]:


parameters = { #This is a dictionary, a more convinient way to pass and access parameters by their names
    'basal_N':1,
    'basal_G':1,# 0.5/20,
    'basal_E': 1, # E parameter to optimise
    'wf_G' : 4,
    'wf_N' : 4,
    'wf_E' : 12, # E parameter to optimise
    'tau_G' : 1,
    'tau_N' : 1,
    'tau_E' : 1,
    'K_GG' : 1.2,
    'K_NN' : 1.2,
    'K_GN' : 1.2,
    'K_NG' : 1.2,
    'K_FN' : 1, #1
    'K_FE' : 3, #3 # E parameter to optimise
    'K_EN' : 1.2, # E parameter to optimise
    'K_NE' : 1.2, # E parameter to optimise
    'K_NEG' : 1.2,
    'K_EG' : 1.2, # E parameter to optimise
    'h' : 4, # could possibly be lowered??
    'FGF' : 0.9, # we will be varying this parameter below.
    'scaleMutualRepression' : 3.5
}

def E_free_func(var, p):
    x = var[0] 
    y = var[1]
    z = var[2]
    
    return z/(1+y/p['K_NEG'])

def E_free_func_t(var_t, p):
    x = var_t[0,:] 
    y = var_t[1,:]
    z = var_t[2,:]
    
    return z/(1+y/p['K_NEG'])


#file_label 3
def equations_NGE_and_F(t, var, p): #we will specify all our equations in one function
    x = var[0] 
    y = var[1]
    z = var[2]
    E_free = E_free_func(var,p)
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(x/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h'])/(1+(y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - y/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(y/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((x/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(x/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - x/p['tau_G']
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((y/p['K_NE'])**p['h'])/((1+(y/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
   
    dEdt = basal_term_E + p['wf_E'] * activation_E_by_N - z/p['tau_E']
    
    return dGdt, dNdt, dEdt

#file_label 2
def equations_NG_and_F(t, var, p): #we will specify all our equations in one function
    var[2]=0
    x = var[0] 
    y = var[1]
    z = var[2]
    E_free = E_free_func(var,p) #((z/p['K_EG'])**p['h'])/(1+(y/p['K_NE'])**p['h']) # I would keep E_free as concentration, but of course shouldn't mateer where we divide by K_EG...

    scaleEar_G = 1
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(x/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h'])/(1+(y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - y/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(y/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((x/(scaleEar_G*p['K_GG']))**p['h'] + (E_free/p['K_EG'])**p['h'])/(1+(x/(scaleEar_G*p['K_GG']))**p['h']+(E_free/p['K_EG'])**p['h']) 
    
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * scaleEar_G*activation_G_by_G_E  - x/p['tau_G']
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((y/p['K_NE'])**p['h'])/((1+(y/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
    
    #For GNF network only:
    dEdt = 0
    
    return dGdt, dNdt, dEdt

#file_label = 4
def equations_NGEcE_and_F(t, var, p): #we will specify all our equations in one function
    x = var[0] 
    y = var[1]
    z = var[2]
    E_free = E_free_func(var,p)#((z/p['K_EG'])**p['h'])/(1+(y/p['K_NE'])**p['h']) # I would keep E_free as concentration, but of course shouldn't mateer where we divide by K_EG...
    
    basal_term_N =  p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))
    inhibition_N_by_G =  (1/(1+(x/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h']))
    activation_N_by_N_E =  ((y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h'])/(1+(y/p['K_NN'])**p['h']+(z/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h'])
    
    dNdt = basal_term_N  + p['scaleMutualRepression']* p['wf_N'] * inhibition_N_by_G * activation_N_by_N_E - y/p['tau_N'] 
   
    inhibition_G_by_N = 1/(1+(y/(p['scaleMutualRepression']*p['K_NG']))**p['h'])
    activation_G_by_G_E = ((x/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h'])/(1+(x/p['K_GG'])**p['h']+(E_free/p['K_EG'])**p['h']) 
    dGdt =  p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * inhibition_G_by_N * activation_G_by_G_E  - x/p['tau_G']
    
    constitutive_E = 1
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((y/p['K_NE'])**p['h'])/((1+(y/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
   
    dEdt = constitutive_E + basal_term_E + p['wf_E'] * activation_E_by_N - z/p['tau_E']
    
    return dGdt, dNdt, dEdt

# In[3]:


def sample_vector_field_v2(equations, variables_grid, parameters, E_fix = 0):
    ''' 
    * input: 
    varaibles_grid is a (2, N, N) array, e.e with NxN x and NxN y. These will be combined in nested loops to cover all combiantions of x and y
    * output: derivs - same shape as variables_grid, but with derivatives for respctive points 
    '''
    g_shape = np.shape(variables_grid)
    (_,N,M) = np.shape(variables_grid)
    print(g_shape)
    derivs_g = np.zeros(g_shape)
    print('derivs_g shape=', np.shape(derivs_g))
    t = 0
    X = np.array(variables_grid[0])
    Y = np.array(variables_grid[1])
    derivs = []
    for i in range(N): #these nested loops is to go through all combinations of X and Y in eh grid
        for j in range(N):
            #derivs = equations(t,np.array([X[i,j], Y[i,j], E_fix]), parameters)
            derivs_g[:,i,j] = equations(t,np.array([X[i,j], Y[i,j], E_fix]), parameters)[:2]#take first two outputs, i.e. DG and DN
            
    #print(np.shape(derivs_g[0]))
    return derivs_g[0], derivs_g[1]


# ## Streamplots and gradients

# In[4]:

def make_streamplot(equations, min_var = 0,  max_var = 15 ,var_step = 1,  E_slice=1):

    #var_step = 1#0.01 #1
    X = np.arange(min_var, max_var, var_step)
    Y = np.arange(min_var, max_var, var_step)

    #1. make the grid for streaplot
    var_grid = np.meshgrid(X,Y)#(min_var, max_var, var_step,min_var, max_var, var_step,min_var, max_var, var_step)

    xdot,ydot = sample_vector_field_v2(equations, var_grid, parameters, E_slice)
    
    #'magma' is a good colormap
    
    fig = plt.figure(figsize=(8,6.5))
    strm = plt.streamplot(var_grid[0], var_grid[1], xdot, ydot, density=3, color=(np.hypot(xdot,ydot)), cmap = 'magma')
    plt.title("Streamplot")
    plt.xlabel('Gata6 Concentration')
    plt.ylabel('Nanog Concentration')
    #plt.clim(0,2) #2.2857
    fig.colorbar(strm.lines)
    
    

def make_streamplots_and_gradients(equations, min_var = 0,  max_var = 15 ,var_step = 1,  E_slice=1):

    var_step = 0.01 #1 
    X = np.arange(min_var, max_var, var_step)
    Y = np.arange(min_var, max_var, var_step)

    #1. make the grid for streaplot
    var_grid = np.meshgrid(X,Y)#(min_var, max_var, var_step,min_var, max_var, var_step,min_var, max_var, var_step)

    xdot,ydot = sample_vector_field_v2(equations, var_grid, parameters, E_slice)

    fig = plt.figure(figsize=(8,6.5))
    strm = plt.streamplot(var_grid[0], var_grid[1], xdot, ydot, density=5, color=(np.hypot(xdot,ydot)), cmap = 'cividis')
    plt.title("Streamplot")
    plt.xlabel('Gata6 Concentration')
    plt.ylabel('Nanog Concentration')
    fig.colorbar(strm.lines)
    
    #################################################
    ##Quiver with color:
    #NB: important that linspaces are defided with low number of points, eg. (0,2.5,40)
    color_array = np.hypot(xdot,ydot)
    #color_map = plt.cm.get_cmap('jet')
    #color = color_map.reversed()
    #'jet' is a good colormap
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    im = ax.quiver(var_grid[0], var_grid[1], xdot, ydot, color_array, cmap = 'tab20b')
    #ss = ax.plot(xdot < 0.01, ydot < 0.01, 'r*')
    plt.title("Quiver with color")
    plt.xlabel('Gata6 Concentration')
    plt.ylabel('Nanog Concentration')
    im.set_clim(0,2)
    fig.colorbar(im, shrink = 1)
    plt.show()
    
    #################################################
    ## Landscape plot, i.e. gradients of the "forces" dV/dr
    Z = np.hypot(xdot,ydot) #Idea from quiver plot coloring
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    imc = ax.contour(var_grid[0], var_grid[1], Z, 80, cmap='jet')
    plt.title("Countour plot of vector magnitudes - forces dV/dr")
    plt.xlabel('Gata6 Concentration')
    plt.ylabel('Nanog Concentration')
    fig.colorbar(imc, shrink = 1)
    #ss = ax.plot(Z < 0.001, Z < 0.001, 'bo')
    #ss = ax.plot(xdot < 0.01, ydot < 0.01, 'bo')
    plt.show()
    
    
    #################################################
    # same as above, but log-scaled
    fig, ax = plt.subplots(figsize=(5.5,4.5))
    imc = ax.contour(var_grid[0], var_grid[1], np.log(Z), 60, cmap='jet')
    plt.title("log normalised countour plot of vector magnitudes - forces dV/dr")
    plt.xlabel('Gata6 Concentration')
    plt.ylabel('Nanog Concentration')
    plt.show()

    #################################################
    # same as above, but contour3D

    fig, ax = plt.subplots(figsize=(11,9))
    ax = plt.axes(projection='3d')

    imc = ax.contour3D(var_grid[0], var_grid[1],np.log(Z),40, cmap='jet')
    ax.view_init(10,50) #angles view

    ax.set_xlabel('Gata6 concentration')
    ax.set_ylabel('Nanog concentration')
    ax.set_zlabel('log(vector magnitude)')
    ax.set_title('3D contour of vector magnitudes, log normalised')

    ax.xaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "black"})
    ax.yaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "black"})
    ax.zaxis._axinfo["grid"].update({"linewidth":0.1, "color" : "black"})

    #plt.axis('off')
    #plt.grid(b=None)
    plt.show()


# In[ ]:





# # Scan initial conditions, bifurcation plots and quntify basin of attraction

# In[5]:

def find_steady_state(ic, parameters, equations, tmax = 150, verbose = False):
    ''' Integrates ODE's specified by "equations" and returns values of variables at "tmax"'''
    sol = solve_ivp(equations, (0,tmax), ic, args = [parameters], max_step = 0.01) # I use exisiting solvers, think they can be faster than euler
    if verbose: #this just gives us an option to check the solutions
        plt.figure()
        print(np.shape(sol.y[0,:]))
        
        plt.plot(sol.t, sol.y[0,:])
        plt.plot(sol.t, sol.y[1,:])
        plt.plot(sol.t, sol.y[2,:])
        plt.plot(sol.t, E_free_func_t(sol.y,parameters))
        
        plt.ylabel('Gata')
        plt.xlabel('t')
        plt.ylim(0,20)
        plt.legend(['Gata', 'Nanog', 'Esrrb', 'E_free'])
    return sol.y[:,-1] #returning gata and nanog for the last time points


# In[6]:


def steadystates_vs_FGF_v2(ic,parameters, equations,FGF_list, tmax=150):
    '''Given initial conditions "ic" (nvar,1) and the list with FGF values, 
    return: table with columns: FGF, var_1(tmax), var2(tmax), var3(tmax), ic1, ic2, ic3 ... '''
    ss = []
    table = np.array([]).reshape(0,7)
    for FGF in FGF_list:
        parameters['FGF'] = FGF
        steady_state = find_steady_state(ic,parameters,equations, tmax, verbose = False)
        table = np.concatenate((table, np.array([FGF, steady_state[0], steady_state[1], steady_state[2], ic[0], ic[1], ic[2]]).reshape(1,7)),axis = 0)
    return table

def steadystates_vs_FGF_v3(ic,parameters, equations,FGF_list, tmax=150):
    '''Given initial conditions "ic" (nvar,1) and the list with FGF values, 
    return: table with columns: FGF, var_1(tmax), var2(tmax), var3(tmax), ic1, ic2, ic3 ... '''
    ss = []
    table = np.array([]).reshape(0,8)
    for FGF in FGF_list:
        parameters['FGF'] = FGF
        steady_state = find_steady_state(ic,parameters,equations, tmax, verbose = False)
        table = np.concatenate((table, np.array([FGF, steady_state[0], steady_state[1], steady_state[2], ic[0], ic[1], ic[2], E_free_func(steady_state, parameters)]).reshape(1,8)),axis = 0)
    return table

# In[7]:



def scan_ic(parameters, equations, FGF_list, N_points=20, tmax = 150, ic_range = [1e-1,3*1e1]):
    ''' With a givne list of FGF values, scans N_points initial ocnditions picked up randomly
    output: tables iwth columns: FGF, var1, var2, var3, ic1, ic2, ic3'''
    min_var = ic_range[0]
    max_var = ic_range[1]
    
    g = loguniform.rvs(min_var, max_var, size=N_points)# generates an array with N_points logunifrom random numbers 
    n = loguniform.rvs(min_var, max_var, size=N_points)# this is to genrate initial conditions on log-space
    e = loguniform.rvs(min_var, max_var, size=N_points)
    
    table_all_ic = np.array([]).reshape(0,7)
    for i in range(N_points): # number of random ic  
        if(i%10==1):
            print(i)
        ic = [g[i],n[i],e[i]] #massage initial conditions into a right shape, 
        table_1 = steadystates_vs_FGF_v2(ic,parameters, equations, FGF_list, tmax) # for given ic, find steady states for all values of FGF in the FGF list
        table_all_ic  = np.concatenate((table_all_ic,table_1), axis = 0) #add FGFs as well as steady states to one big tale. Can evt. also keep track of i.c. here
    return table_all_ic
    
# In[8]:


def scan_ic_v2(parameters, equations, FGF_list,  ics, tmax = 150):
    ''' With a givne list of FGF values, scans the (N_points, 3) array of initial conditions "ics" 
    output: tables iwth columns: FGF, var1, var2, var3, ic1, ic2, ic3'''

    table_all_ic = np.array([]).reshape(0,7)
    (N_points,_) = np.shape(ics) 
    for i in range(N_points): # number of random ic  
        if(i%10==1): # this is to print every 10th step to see progress
            print(i)
        ic = ics[i,:] #massage initial conditions into a right shape,
        table_1 = steadystates_vs_FGF_v2(ic,parameters, equations, FGF_list, tmax) # for given ic, find steady states for all values of FGF in the FGF list
        table_all_ic  = np.concatenate((table_all_ic,table_1), axis = 0) #add FGFs as well as steady states to one big tale. Can evt. also keep track of i.c. here
    return table_all_ic

def scan_ic_v3(parameters, equations, FGF_list,  ics, tmax = 150):
    ''' With a givne list of FGF values, scans the (N_points, 3) array of initial conditions "ics" 
    output: tables iwth columns: FGF, var1, var2, var3, ic1, ic2, ic3'''

    table_all_ic = np.array([]).reshape(0,8)
    (N_points,_) = np.shape(ics) 
    for i in range(N_points): # number of random ic  
        if(i%10==1): # this is to print every 10th step to see progress
            print(i)
        ic = ics[i,:] #massage initial conditions into a right shape,
        table_1 = steadystates_vs_FGF_v3(ic,parameters, equations, FGF_list, tmax) # for given ic, find steady states for all values of FGF in the FGF list
        table_all_ic  = np.concatenate((table_all_ic,table_1), axis = 0) #add FGFs as well as steady states to one big tale. Can evt. also keep track of i.c. here
    return table_all_ic



def generate_ic_to_scan_random(ic_range = [1e-1,3*1e1], N_points=10):
    '''Returns a (N_points,3) array with IC picked from loguniform distribution'''
    ic = loguniform.rvs(ic_range[0], ic_range[1], size=(N_points,3))# generates an array with N_points logunifrom random numbers
    return ic

def generate_ic_to_scan_deterministic_v2(ic_range = [1e-1,3*1e1], N_points_1D=10, base = 2):
    '''Returns a (N_points,3) array with IC picked on a grid'''
    start_lin = ic_range[0]
    end_lin = ic_range[1]

    start_log = np.log(start_lin)/np.log(base) #ln
    end_log = np.log(end_lin)/np.log(base)
    nx,ny,nz = (N_points_1D, N_points_1D, N_points_1D)
    x = np.logspace(start_log,end_log,nx, base = base)
    y = np.logspace(start_log,end_log,ny, base = base)
    z = np.logspace(start_log,end_log,nz, base = base)
    
    xv, yv, zv = np.meshgrid(x, y,z) # this creates a grid , in the next couple of lines we massage this grdi into the same shape as we used when generated ic randomly
    ic = np.empty((nx*ny*nz,3)) #preallocating space, in this case the N_points = nx*ny*nz, so = N_points_1D^3
    count = 0
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ic[count, :] = [xv[i,j,k], yv[i,j,k], zv[i,j,k]]
                count+=1

    return ic

# In[8b]:
def find_ss_E(x,y, p):
    #x = var[0] 
    #y = var[1]
    
    basal_term_E =  (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h']))
    activation_E_by_N = ((y/p['K_NE'])**p['h'])/((1+(y/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])) 
    EF = (basal_term_E + p['wf_E'] * activation_E_by_N) * p['tau_E'] 
    #print(EF)
    return EF

def lin_to_log(x):
    return np.log2(x)


def generate_ic_to_scan_deterministic_triangle(p, ic_range = [0.1,50], N_points_1D=10, base = 2, file_label=0):
    '''Returns a (N_points,3) array with IC picked on a grid'''
    start_lin = ic_range[0]
    end_lin = ic_range[1]

    start_log = np.log2(start_lin)
    end_log = np.log2(end_lin)
    nx,ny,nz = (N_points_1D, N_points_1D, N_points_1D)
    x = np.logspace(start_log,end_log,nx, base = base)
    y = np.logspace(start_log,end_log,ny, base = base)
    z = 0
    
    def calc_z(file_label, xv,yv):
        z = 0
        if file_label == 0:
            print("No file_label specified")
        elif file_label == 2:
            z = 0
        elif file_label == 3:
            z = find_ss_E(xv,yv,p)
        else:
            print("file_label > 3")
        return z

    xv, yv = np.meshgrid(x, y) # this creates a grid , in the next couple of lines we massage this grdi into the same shape as we used when generated ic randomly
    ic = []
    ic = np.empty((0,3)) #preallocating space, in this case the N_points = nx*ny*nz, so = N_points_1D^3
    count = 0
    for i in range(nx):
        for j in range(ny):
            if yv[i,j] <= - (end_lin/end_lin)*xv[i,j] + end_lin:
                z = calc_z(file_label,xv[i,j],yv[i,j])
                ic = np.concatenate((ic, np.array([xv[i,j], yv[i,j], z]).reshape(1,3)), axis=0)
                count+=1                
    return ic


def generate_ic_to_scan_random_triangle(p, ic_range = [0.1,50], N_points=1000, base = 2, file_label=0):
    '''Returns a (N_points,3) array with IC picked on a grid'''
    random.seed(12345)
    ic_start = loguniform.rvs(ic_range[0], ic_range[1], size=(N_points,2))# generates an array with N_points logunifrom random numbers
    
    def calc_z(file_label, xv,yv):
        z = 0
        if file_label == 0:
            print("No file_label specified")
        elif file_label == 2:
            z = 0
        elif file_label == 3:
            z = find_ss_E(xv,yv,p)
        else:
            print("file_label > 3")
        return z

    ic = []
    ic = np.empty((0,3)) #preallocating space, in this case the N_points = nx*ny*nz, so = N_points_1D^3
    end_lin = ic_range[1]
        
    for i in range(N_points):
        
        x = ic_start[i,0]
        y = ic_start[i,1]
        if y <= - (end_lin/end_lin)*x + end_lin:
            z = calc_z(file_label,x,y)
            ic = np.concatenate((ic, np.array([x, y, z]).reshape(1,3)), axis=0)
    return ic

# In[9]:

def plot_bifurcation(table_all_ic):
    '''Takes the tbale with columns FGF, var1(tmax), var2(tmax) ... and plots the var(tmax) vs FGF'''
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize = (12,4))
    fig.suptitle('bifurcation')
    
    ax1.semilogx(table_all_ic[:,0],table_all_ic[:,1],'*r', markersize = 3, alpha = 0.2)
    ax1.set(xlabel = 'FGF', ylabel = 'Gata')
    ax1.set_ylim(-1,18)
    #ax1.set_xticks(ticks = np.logspace(-1,1,10, base = 2 ), minor=False)

    ax2.semilogx(table_all_ic[:,0],table_all_ic[:,2],'*g', markersize = 3, alpha = 0.2)
    ax2.set(xlabel = 'FGF', ylabel = 'Nanog')
    
    ax3.semilogx(table_all_ic[:,0],table_all_ic[:,3],'*b', markersize = 3, alpha = 0.2)
    ax3.set(xlabel = 'FGF', ylabel = 'Esrrb')
    

def plot_bifurcation_v3(table_all_ic):
    '''Takes the tbale with columns FGF, var1(tmax), var2(tmax) ... and plots the var(tmax) vs FGF'''
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True, figsize = (15,4))
    fig.suptitle('bifurcation')
    
    ax1.semilogx(table_all_ic[:,0],table_all_ic[:,1],'*r', markersize = 3, alpha = 0.2)
    ax1.set(xlabel = 'FGF', ylabel = 'Gata')
    ax1.set_ylim(-1,18)
    #ax1.set_xticks(ticks = np.logspace(-1,1,10, base = 2 ), minor=False)

    ax2.semilogx(table_all_ic[:,0],table_all_ic[:,2],'*g', markersize = 3, alpha = 0.2)
    ax2.set(xlabel = 'FGF', ylabel = 'Nanog')
    
    ax3.semilogx(table_all_ic[:,0],table_all_ic[:,3],'*b', markersize = 3, alpha = 0.2)
    ax3.set(xlabel = 'FGF', ylabel = 'Esrrb')
    
    ax4.semilogx(table_all_ic[:,0],table_all_ic[:,-1],'*b', markersize = 3, alpha = 0.2)
    ax4.set(xlabel = 'FGF', ylabel = 'Esrrb_free')
   
    

# In[10]:


def find_basin(fgf_range,gata_range, table_all_ic):
    '''Returns:
    1: number of points in requested ranges
    2: fraction of points in requested ranges
    3: the initial conditions, that resulted in the points in a ginven range'''
    
    def find_values_in_FGF_range(FGF_range, table_all_ic):
        ''' FGF_range = [fgf_min, fgf_max]
        '''
        FGF = table_all_ic[:,0]
        #print(np.shape(FGF))
        mask = (FGF>=FGF_range[0]) & (FGF<=FGF_range[1]) # mask returning true or false for FGF satisfying the conditions
        idx, = np.where(mask) # find indices where mask has values =True

        return table_all_ic[idx,:] # return subtable with FGFs withint a givne range

    def find_values_in_Gata_range(gata_range, table):
        ''' gata_range = [gata_min, gata_max]
        '''
        gata = table[:,1]
        #print(np.shape(gata))
        #print(gata)
        mask = (gata>=gata_range[0]) & (gata<=gata_range[1]) # mask returning true or false for FGF satisfying the conditions
        idx, = np.where(mask) # find indices where mask has values =True

        return table[idx,:] # return subtable with FGFs withint a givne range

    
    table_in_FGF_range = find_values_in_FGF_range(fgf_range, table_all_ic)#, [1,8], table_all_ic)
    table_in_FGF_Gata_range = find_values_in_Gata_range(gata_range, table_in_FGF_range)
    
    (number_of_points_in_range,_) =np.shape(table_in_FGF_Gata_range) 
    (total_points,_)= np.shape(table_all_ic) 
    #ic_in_range = table_in_FGF_Gata_range[:,4:]
    
    return number_of_points_in_range, number_of_points_in_range/total_points, table_in_FGF_Gata_range


# In[11]:
def find_basin_Nanog(fgf_range,Nanog_range, table_all_ic):
    '''Returns:
    1: number of points in requested ranges
    2: fraction of points in requested ranges
    3: the initial conditions, that resulted in the points in a ginven range'''
    
    def find_values_in_FGF_range(FGF_range, table_all_ic):
        ''' FGF_range = [fgf_min, fgf_max]
        '''
        FGF = table_all_ic[:,0]
        #print(np.shape(FGF))
        mask = (FGF>=FGF_range[0]) & (FGF<=FGF_range[1]) # mask returning true or false for FGF satisfying the conditions
        idx, = np.where(mask) # find indices where mask has values =True

        return table_all_ic[idx,:] # return subtable with FGFs withint a givne range

    def find_values_in_Nanog_range(Nanog_range, table):
        ''' Nanog_range = [Nanog_min, Nanog_max]
        '''
        Nanog = table[:,2]
        #print(np.shape(gata))
        #print(gata)
        mask = (Nanog>=Nanog_range[0]) & (Nanog<=Nanog_range[1]) # mask returning true or false for FGF satisfying the conditions
        idx, = np.where(mask) # find indices where mask has values =True

        return table[idx,:] # return subtable with FGFs withint a givne range

    
    table_in_FGF_range = find_values_in_FGF_range(fgf_range, table_all_ic)#, [1,8], table_all_ic)
    table_in_FGF_Nanog_range = find_values_in_Nanog_range(Nanog_range, table_in_FGF_range)
    
    (number_of_points_in_range,_) =np.shape(table_in_FGF_Nanog_range) 
    (total_points,_)= np.shape(table_all_ic) 
    #ic_in_range = table_in_FGF_Gata_range[:,4:]
    
    return number_of_points_in_range, number_of_points_in_range/total_points, table_in_FGF_Nanog_range

def sample_trajectories(ic, parameters, equations, tmax_int = 50, tmax_ss = 200, verbose = False, E_slice = 0):
    ''' Integrates ODE's specified by "equations" and returns values of variables at "tmax"'''
    t_eval = np.linspace(0,200,100)
    sol = solve_ivp(equations, (0,tmax_int),  ic, args = [parameters])
    sol_ss = solve_ivp(equations, (0,tmax_ss), ic, t_eval = t_eval, args = [parameters]) 
    if verbose: #this just gives us an option to check the solutions
        min_var = 0
        max_var = 20
        var_step = 1
        
        plt.figure(figsize=(15,5))
        
        plt.subplot(1, 2, 1)
        plt.plot(sol_ss.y[0,:],sol_ss.y[1,:], marker='o', linestyle='dashed') #sol.y[0,:],sol.y[1,:]
        plt.plot(sol.y[0,-1],sol.y[1,-1], marker='o', color = 'orange')
        plt.plot(sol_ss.y[0,-1],sol_ss.y[1,-1], marker='o', color = 'red')
        plt.ylabel('Nanog')
        plt.xlabel('Gata6')
        plt.xlim(0,20)
        plt.ylim(0,20)
        
        plt.subplot(1, 2, 2)
        plt.plot(sol_ss.y[1,:], sol_ss.y[2,:], marker='o', linestyle='dashed')
        plt.plot(sol.y[1,-1],sol.y[2,-1], marker='o', color = 'orange')
        plt.plot(sol_ss.y[1,-1],sol_ss.y[2,-1], marker='o', color = 'red')
        plt.xlabel('Nanog')
        plt.ylabel('EsrrB')
        plt.xlim(0,20)
        plt.ylim(0,20)    

        #make_streamplot(equations,min_var, max_var, var_step, E_slice = E_slice)
        
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(sol.y[1,:], sol.y[0,:], sol.y[2,:])
        plt.ylabel('Gata')
        plt.xlabel('Nanog')
        #plt.zlabel('EsrrB')
        
        
        p = ax.scatter3D(sol.y[1,:], sol.y[0,:], sol.y[2,:], c=sol.y[0,:], cmap='Greens');

        fig.colorbar(p)
        #plt.ylim(0,20)
        #plt.legend(['Gata', 'Nanog', 'Esrrb', 'E_free'])
    return sol.y #returning gata and nanog for the last time points



def sample_time_to_ss(ic, parameters, equations, tmax = 150, verbose = False):
    ''' Integrates ODE's specified by "equations" and returns the time it takes for system to get to '''
    
    teval = np.linspace(0,tmax,100)
    sol = solve_ivp(equations, (0,tmax), ic, args = [parameters])
    # I use exisiting solvers, think they can be faster than euler
    
    def find_tss(sol):
        t = 0
        count = 0
        dcount = 5
        slope = 10
        ss_threshold = 0.01
        while sol.t[count] < sol.t[-1] and slope>ss_threshold:
            if count-dcount>=0: # make sure that the prev element exists
                slope = np.max((sol.y[:2, count]- sol.y[:2, count - dcount])/sol.y[:2, count])
            count +=1
        return count-1, sol.t[count]
    
    countss, tss = find_tss(sol)
    
    if verbose: #this just gives us an option to check the solutions
        plt.figure(figsize = (15,5))
        labels = ['Gata', 'Nanog', 'EsrrB']
        
        for i in range(3):
            plt.subplot(1, 3, i+1)
            plt.plot(sol.t, sol.y[i,:])
            plt.plot(sol.t[countss], sol.y[i,countss], '*')
            plt.ylabel(labels[i])
            plt.xlabel('t')

    return tss

def sample_time_to_ss_v2(ic, parameters, equations, FGF, tmax = 200):
    ''' Integrates ODE's specified by "equations" and returns the time it takes for system to get to '''
    
    teval = np.linspace(0,tmax,100)
    parameters['FGF'] = FGF
    sol = solve_ivp(equations, (0,tmax), ic, args = [parameters])
    # I use exisiting solvers, think they can be faster than euler
    
    def find_tss(sol):
        t = 0
        count = 0
        dcount = 5
        slope = 10
        ss_threshold = 0.01
        while sol.t[count] < sol.t[-1] and slope>ss_threshold:
            if count-dcount>=0: # make sure that the prev element exists
                slope = np.max((sol.y[:2, count]- sol.y[:2, count - dcount])/sol.y[:2, count])
            count +=1
        return count-1, sol.t[count]
    
    countss, tss = find_tss(sol)
    return tss

def scan_ic_timess(parameters, equations, FGF,  ics, tmax = 200, file_label = 0):
    ''' With a givne list of FGF values, scans the (N_points, 3) array of initial conditions "ics" 
    output: tables iwth columns: FGF, var1, var2, var3, ic1, ic2, ic3'''
    
    table_all = np.array([]).reshape(0,6)
    (N_points,_) = np.shape(ics)
    for i in range(N_points): # number of random ic
        row = np.array([]).reshape(0,6)
        if(i%100==1): # this is to print every 10th step to see progress
            print(i)
        ic = ics[i,:] #massage initial conditions into a right shape,
        tss = sample_time_to_ss_v2(ic,parameters, equations, FGF, tmax)
        row1 = np.concatenate((row, np.array([FGF, ic[0], ic[1], ic[2], tss, file_label]).reshape(1,6)),axis = 0)
        table_all = np.concatenate((table_all,row1), axis = 0)
    return table_all