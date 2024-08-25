import torch
import time

class POD_ODE_Solver:

    def __init__(self, C, G, P, time_steps, num_modes, dt, sampling_interval, multiple=True):
        '''
        ODE solver for c(db/dt) + Gb = P
        multiple: whether or not solving all functional units in parallel
        Requires for each functional unit i:
            C: num_modes_i x num_modes_i
            G: num_modes_i x num_modes_i
            P: num_modes_i x time_steps
        If using multiple functional units all at once, C, G, P needs to be lists of tensors
        '''
        self.C = C
        self.G = G
        self.P = P
        self.time_steps = time_steps
        if multiple:
            self.num_modes = sum(num_modes)
        else:
            self.num_modes = num_modes
        self.dt = dt
        self.device = C.device
        self.sampling_interval = sampling_interval
        self.multiple = multiple
    
    def initial_conditions(self):
        a_0 = torch.zeros(self.num_modes).to(self.device).double()
        return a_0
    
    def solve(self):
        #set up coefficients for ode
        if self.multiple:
            CG_list = []
            CP_list =[]
        for i in len(self.C):
            C_temp = self.C[i].t()
            C_temp = C_temp[i].t()
            G_temp = self.G[i].t()
            G_temp = G_temp[i].t()
            P_temp = self.P[i]

            C_inverse = torch.linalg.inv(C_temp)
            CG = C_inverse.matmul(G_temp)
            for j in range(40):
                CG_list.append(CG)

            CP = (C_inverse).matmul(P_temp)
            for j in range(40):
                CP_list.append(CP)

        CG = torch.block_diag(*CG_list)
        CP = torch.cat(CP_list, dim=0)
        CP = CP.t()
        C_inverse = torch.linalg.inv(self.C)
        CG = C_inverse.matmul(self.G)
        CP = (C_inverse).matmul(self.P)
        CP = CP.t()
        #solution vectors
        u = torch.zeros((self.time_steps + 1, self.num_modes)).to(self.device).double()
        #current time solution
        a_n = self.initial_conditions() 
        u[0] = a_n
        initial = torch.zeros_like(CP[0]).double().to(CP.device)
        for i in range(0, self.time_steps):
            #implement RK4
            a_n = u[i]
            x1 = initial if i == 0 else CP[i - 1]
            x2 = CP[i]
            curr_arg = x1
            arg_step = (x2 - x1) / 20
            for j in range(20):
                interp_arg = curr_arg + arg_step
                next_arg = interp_arg + arg_step
                k1 = self.dt * self.f(curr_arg, CG, a_n)
                k2 = self.dt * self.f(interp_arg, CG, a_n + k1 / 2)
                k3 = self.dt * self.f(interp_arg, CG, a_n + k2 / 2)
                k4 = self.dt * self.f(next_arg, CG, a_n + k3)
                a_vec = a_n + (k1 + 2 * k2 + 2 * k3 + k4) / 6
                a_n = a_vec
                curr_arg = next_arg
            u[i + 1] = a_n
        return u[1:]
    
    
    def f(self, arg, CG, a):
        #ODE function
        return arg - CG.matmul(a)


