import numpy as np

def RK4_np(f):
    return lambda t, y, dt: (
            lambda dy1: (
            lambda dy2: (
            lambda dy3: (
            lambda dy4: (dy1 + 2*dy2 + 2*dy3 + dy4)/6
            )( dt * np.array(f( t + dt  , y + dy3   )) )
	    )( dt * np.array(f( t + dt/2, y + dy2/2 )) )
	    )( dt * np.array(f( t + dt/2, y + dy1/2 ) ))
	    )( dt * np.array(f( t       , y         ) ))

def Runge_Kutta_run_v1(System, initial_state, timestop = 1000, dt = 0.01):
    CurSystem = System
    dy_np = RK4_np(lambda t,y: CurSystem.dx(t,y))
    
    while CurSystem.time[-1] < timestop:
        
        #CurSystem.ERK[-1] = Erk_func(CurSystem.time[-1])
                
        # Get current state
        N = CurSystem.N[-1]
        E = CurSystem.E[-1]
        G = CurSystem.G[-1]
        #ERK = CurSystem.ERK[-1]

        # Do a runge-kutta step
        result = dy_np(t = 0, y = [N,E,G],dt = dt)

        # Save result
        CurSystem.time.append(CurSystem.time[-1] + dt)
        CurSystem.N.append(CurSystem.N[-1] + result[0])
        CurSystem.E.append(CurSystem.E[-1] + result[1])
        CurSystem.G.append(CurSystem.G[-1] + result[2])
       # CurSystem.ERK.append(CurSystem.ERK[-1] + result[3])
        
    CurSystem.RK_version = '2'
    return CurSystem