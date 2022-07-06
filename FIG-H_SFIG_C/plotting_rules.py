from matplotlib import gridspec
import numpy as np
import matplotlib.pylab as plt


def prepare_for_plotting(GillespieSystem):
    time_spent_in_state = np.array(GillespieSystem.time)[1:-1]-np.array(GillespieSystem.time)[0:-2]
    G_state = np.array(GillespieSystem.G)[0:-2]
    N_state = np.array(GillespieSystem.N)[0:-2]

    N_low = np.zeros(max(GillespieSystem.N)+1)
    N_high = np.zeros(max(GillespieSystem.N)+1)
    N_all = np.zeros(max(GillespieSystem.N)+1)
    for i, number in enumerate(N_state):
        if N_state[i] >= G_state[i]: 
            N_high[number] = N_high[number] + time_spent_in_state[i]
        if N_state[i] < G_state[i]: 
            N_low[number] = N_low[number] + time_spent_in_state[i]
        N_all[number] = N_all[number] + time_spent_in_state[i]



    G_low  = np.zeros(max(GillespieSystem.G)+1)
    G_high = np.zeros(max(GillespieSystem.G)+1)
    G_all  = np.zeros(max(GillespieSystem.G)+1)
    for i, number in enumerate(G_state):
        if G_state[i] >= N_state[i]: 
            G_high[number] = G_high[number] + time_spent_in_state[i]
        if G_state[i] < N_state[i]: 
            G_low[number] = G_low[number] + time_spent_in_state[i]
        G_all[number] = G_all[number] + time_spent_in_state[i]
        
    Bar_all  = np.zeros((max(GillespieSystem.N)+1,max(GillespieSystem.G)+1))
    for i in range(len(time_spent_in_state)):
        Bar_all[N_state[i],G_state[i]] += time_spent_in_state[i]
    
    return N_state, N_low, N_high, N_all, G_state, G_low, G_high, G_all, time_spent_in_state, Bar_all
        
def GillespiePlotter(GillespieSystem, plot_every_nth_point = 1, figsize = (8,6), ylim = [0,150]):
    N_state, N_low, N_high, N_all, G_state, G_low, G_high, G_all, time_spent_in_state, Bar_all = prepare_for_plotting(GillespieSystem)

    fig = plt.figure(figsize = figsize)
    title = ', paramset:' + GillespieSystem.param_set
    st = fig.suptitle(title, fontsize="x-large")
    
    gs = gridspec.GridSpec(1, 5)
    ylim_low = ylim[0]
    ylim_high = ylim[1]

    # time series
    ax1 = plt.subplot(gs[0, 0:3])
    plot_timeserie(GillespieSystem,plot_every_nth_point= plot_every_nth_point, figsize=figsize)
    plt.ylim([ylim_low,ylim_high])
    plt.ylabel('# molecules')
    plt.xlabel('time')

    # N hist
    ax2 = plt.subplot(gs[0,3])

    plt.barh(bottom = np.array(range(max(GillespieSystem.N)+1))-0.5 , width  = N_low, height =1, color = 'r', alpha = 0.3, lw = 0)
    plt.barh(bottom = np.array(range(max(GillespieSystem.N)+1))-0.5 , width  = N_high, height =1, color = 'r', alpha = 0.3, lw = 0)
    plt.barh(bottom = np.array(range(max(GillespieSystem.N)+1))-0.5 , width  = N_all, height =1, color = 'r', alpha = 0.3, lw = 0)

    plt.xlim([0,np.max(N_all)])
    plt.ylim([ylim_low,ylim_high])
    plt.xlabel('time in state')
    plt.title('Hist N')
    
    # G hist
    ax3 = plt.subplot(gs[0,4])

    plt.barh(bottom = np.array(range(max(GillespieSystem.G)+1))-0.5 , width  = G_low, height =1, color = 'k', alpha = 0.5, lw = 0)
    plt.barh(bottom = np.array(range(max(GillespieSystem.G)+1))-0.5 , width  = G_high, height =1, color = 'k', alpha = 0.5, lw = 0)
    plt.barh(bottom = np.array(range(max(GillespieSystem.G)+1))-0.5 , width  = G_all, height =1, color = 'k', alpha = 0.5, lw = 0)

    plt.xlim([0,np.max(G_all)])
    plt.ylim([ylim_low,ylim_high])
    plt.xlabel('time in state')
    plt.title('Hist G')
    
    plt.rcParams['figure.figsize'] = figsize
    return fig

def plot_timeserie(CurSystem, plot_every_nth_point = 1, figsize = (8,6), ERK_scaling = 1):
    penp = plot_every_nth_point
    
    plt.figure(figsize = figsize)
    #plt.hold
    #plt.plot(CurSystem.time[0::penp],np.array(CurSystem.ERK[0::penp])/ERK_scaling,'b', label = 'ERK')
    plt.plot(CurSystem.time[0::penp],CurSystem.N[0::penp],'r', label = 'N')
    plt.plot(CurSystem.time[0::penp],CurSystem.E[0::penp],'g', label = 'E')
    plt.plot(CurSystem.time[0::penp],CurSystem.G[0::penp],'k', label = 'G')
    
    plt.title('Timeserie')
    plt.legend()
    plt.axis('tight')
