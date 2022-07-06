import random
import numpy as np
from itertools import accumulate
import bisect

def Gillespie_run_v1(System, initial_state, timestop = 1000,):

    CurSystem = System

    while CurSystem.time[-1] < timestop:

       # CurSystem.ERK[-1] = Erk_func(CurSystem.time[-1])

        # Calculate rates, according to the current state and the staterules for 
        # production and dilution of each molecule
        
        rs = CurSystem.rs_func_values()
        r = sum(rs)

        # Draw two random numbers for the gillespie algorithm to figure out 
        # which even happens
        a_unirand = random.random();
        t_next = -np.log(a_unirand)/r;
        b_unirand = random.random();
        #rs_cumsum = np.cumsum(rs)/r
        rs_cumsum = list(accumulate(rs))
        rs_cumsum = [elem/r for elem in rs_cumsum]
        happens_idx = bisect.bisect_left(rs_cumsum, b_unirand)  
            #Seems faster than next(i for i,v in enumerate(L) if v > 0.7)

        # Update state according to gillespie result 
        CurSystem.gillespie_update_state(CurSystem.rs_name[happens_idx],
                                         CurSystem.rs_pm[happens_idx])
        CurSystem.time.append(CurSystem.time[-1] + t_next);
    
    return CurSystem