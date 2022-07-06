import copy
from math import sqrt
from System_004_param import *

import os
print('loaded: '+ os.path.basename(__file__))

class System_simplified:
    def __init__(self, initial_state, init_time = 0, network_ID = -1, p = parameters):
                
        self.state = initial_state        # INITIAL STATE AS DICT
        self.param_set = param_set
        self.N = copy.copy(initial_state['N'])
        self.network_ID = network_ID
        if self.network_ID == 2: ## always use 0 i.c. for esrrB in 2 node network
            #initial_state['E'] = [0]
            self.E = [0]#copy.copy(initial_state['E'])
            #self.E = copy.copy(initial_state['E'])
        else:
            self.E = copy.copy(initial_state['E'])
        
        self.G = copy.copy(initial_state['G'])
        #self.ERK = copy.copy(initial_state['ERK'])
        
        
        self.names = initial_state.keys() # MOLECULE NAMES
        self.time = [init_time]
        
        self.Ef = lambda N,E : E/(1+N/p['K_NEG'])
        
        # DEFINE RULES FOR THE SYSTEM
        self.state_rules = {}
        for name in initial_state.keys():
            self.state_rules[name] = {}
        
        self.state_rules['N']['plus'] =    lambda N,E,G: p['scaleMutualRepression']*(p['basal_N']/(1 + (p['FGF']/(p['K_FN']))**p['h']))  + p['scaleMutualRepression']* p['wf_N'] * (1/(1+(G/(p['scaleMutualRepression']*p['K_GN']))**p['h']+(p['FGF']/p['K_FN'])**p['h'])) * ((N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h'])/(1+(N/p['K_NN'])**p['h']+(E/p['K_EN'])**p['h']+(p['FGF']/p['K_FN'])**p['h']);
        self.state_rules['N']['minus'] =   lambda N,E,G: N/p['tau_N'] ;
        
        
        self.state_rules['G']['plus'] =    lambda N,E,G: p['basal_G'] + p['scaleMutualRepression']*p['wf_G'] * 1/(1+(N/(p['scaleMutualRepression']*p['K_NG']))**p['h']) * ((G/p['K_GG'])**p['h']+(self.Ef(N,E)/p['K_EG'])**p['h'])/(1+(G/p['K_GG'])**p['h']+(self.Ef(N,E)/p['K_EG'])**p['h']);
        self.state_rules['G']['minus'] =   lambda N,E,G: G/p['tau_G'];
        
        if self.network_ID == 2:
            self.state_rules['E']['plus']  = lambda N,E,G: 0*((p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h'])) + p['wf_E'] * ((N/p['K_NE'])**p['h'])/((1+(N/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h'])));
            self.state_rules['E']['minus'] = lambda N,E,G: 0*E/p['tau_E'];
        else:
            self.state_rules['E']['plus'] =    lambda N,E,G: (p['basal_E']/(1 + (p['FGF']/(p['K_FE']))**p['h'])) + p['wf_E'] * ((N/p['K_NE'])**p['h'])/((1+(N/p['K_NE'])**p['h']+(p['FGF']/p['K_FE'])**p['h']));
            self.state_rules['E']['minus'] =   lambda N,E,G: E/p['tau_E'];
        
        ###################### FOR RUNGE KUTTA
        #self.dERK = lambda N,E,G : self.state_rules['ERK']['plus'](N,E,G ) - self.state_rules['ERK']['minus'](N,E,G );
        self.dN =   lambda N,E,G: self.state_rules['N']['plus'](N,E,G ) - self.state_rules['N']['minus'](N,E,G );
        self.dE=    lambda N,E,G: self.state_rules['E']['plus'](N,E,G ) - self.state_rules['E']['minus'](N,E,G );
        self.dG=    lambda N,E,G: self.state_rules['G']['plus'](N,E,G ) - self.state_rules['G']['minus'](N,E,G );
        self.d_ALL= lambda N,E,G: [self.dN(N,E,G), self.dE(N,E,G), self.dG(N,E,G)]
        
        # FOR GILLESPIE
        self.rs_name = []
        self.rs_pm = []
        self.rs_func = []
        for moleculekey in self.state_rules.keys():
            for p_or_m in self.state_rules[moleculekey].keys():
                self.rs_name.append(moleculekey)
                if p_or_m == 'plus':
                    self.rs_pm.append(1)
                elif p_or_m == 'minus':
                    self.rs_pm.append(-1)
                self.rs_func.append(self.state_rules[moleculekey][p_or_m])    
        self.rs_func_list = lambda N,E,G  : [self.rs_func[i](N,E,G ) for i in range(6)]
    
    def dx(self,t,y):
        N = y[0]
        E = y[1]
        G = y[2]
        return self.dN(N,E,G ), self.dE(N,E,G ), self.dG(N,E,G )
    
    def get_molecule_state_list(self):
        return [self.N[-1], self.E[-1], self.G[-1]]
    
    def get_molecule_state_comma(self):
        return self.N[-1], self.E[-1], self.G[-1]#, self.ERK[-1]
    
    def append_molecule_state(self, newstate):
        self.N.append(newstate[0])
        self.E.append(newstate[1])
        self.G.append(newstate[2])
        #self.ERK.append(newstate[3])
        
    # These might benefit from all being in the same func, so _index wont be called 2 times. See linetimer if neseasery
    
    def rs_func_values(self):
        return self.rs_func_list(self.N[-1], self.E[-1], self.G[-1])
    
    def rs_find_largest_funcindex(self):
        #max_index = np.argmax(self.rs_func_values())
        max_index, max_value = max(enumerate(self.rs_func_values()), key=operator.itemgetter(1))
        return max_index
    
    def rs_find_largest_name(self):
        return self.rs_name[self.rs_find_largest_funcindex()]
    
    def rs_find_largest_pm(self):
        return self.rs_pm[self.rs_find_largest_funcindex()]
    
    def gillespie_update_state(self,name,pm):
        self.N.append(self.N[-1])
        self.E.append(self.E[-1])
        self.G.append(self.G[-1])
        #self.ERK.append(self.ERK[-1])
        
        if name == 'N':
            self.N[-1] = max(self.N[-1] + pm , 1)
        elif name == 'E':
            self.E[-1] = max(self.E[-1] + pm , 1)  
        elif name == 'G':
            self.G[-1] = max(self.G[-1] + pm , 1)
        #elif name == 'ERK':
        #    self.ERK[-1] = max(self.ERK[-1] + pm , 1)
        
#CurSystem = System(initial_state)