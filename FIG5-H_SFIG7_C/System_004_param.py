# SET STARTING BINDING AFFINITIES

import os
print('loaded: '+ os.path.basename(__file__))


param_set = '14032022'

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
    'K_FN' : 1,
    'K_FE' : 3, # E parameter to optimise
    'K_EN' : 1.2, # E parameter to optimise
    'K_NE' : 1.2, # E parameter to optimise
    'K_NEG' : 1.2,
    'K_EG' : 1.2, # E parameter to optimise
    'h' : 4, # could possibly be lowered??
    'FGF' : 0.85, # we will be varying this parameter below.
    'scaleMutualRepression' : 3.5
}


## DEGREDATION
#Tau_N   =   2       # ~2 hours
#Tau_G   =   1.92 #2 # 1.6 #1.92  # from Victor   ##################
#Tau_E   =   8       # lookup...
#Tau_Erk =   0.5
#
## BASAL PRODUCTION
#alpha_N =   0.5
#alpha_E =   0.5
#alpha_G =   0
#
## PRODOCTION SCALE
#beta_N =  800
#beta_E =  150
#beta_G =  200 #75                           ##################
#
## BINDING AFFINITIES
#k_NE   =   75 #450
#k_NErk =   95
#k_NG   =  100
#
#k_EN   =  230
#k_EErk = 1000
#
#k_GE   =  40 # 20 #10                      ##################
#k_GEN  =  1000                               ##################
#
#
## HILL COEFFICIENTS
#h_NE   =    1 # N where E comes and binds to it
#h_EN   =    1 # E where N comes and binds to it
#h_GE   =    2                              ##################

#K_NN = 100;
#K_ND = 100;
#K_NE = 100;
#K_NG = 100;

#K_ErkE = 100;
#K_ErkN = 100;
#K_ErkD = 100;

#K_EdG = 100;

#K_EG = 10;
#K_EN = 0.1;

#K_GG = 100;

## SET SCALING FOR PRODUCTION TERMS
##alpha_1 = 100; # N
#basal_N = 10    # N
#alpha_1 = 50;   # N
#beta_1 = 100;  # D
#basal_E = 5;    # E
#gamma_1 = 10;   # E
#basal_G = 5        # G
#delta_1_yesG = 40; # G
#delta_1_noG = 90;  # G Use this for no Gfeedback
#delta_2 = 0.3;     # G positive autofeedback

## SET DEGREDATION SPEEDS
#Tau_N = 2;
#Tau_E = 24;
#Tau_G = 1.92;
##Tau_G = 8.92;
#Tau_ERK = 0.5;