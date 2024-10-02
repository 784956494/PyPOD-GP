import torch
from fenics import *
from dolfin import * 
from mshr import *
import numpy as np
import time
import h5py
import utils
from config import parser
from pypod_gp import PyPOD_GP
from mpi4py import MPI
import sys
import pandas as pd
from utils import POD_ODE_Solver

def run_prediction(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    flp = np.loadtxt('path to floor plan')
    pds = np.loadtxt('path to power density')
    thick_actl =1.35e-4 
    h_c = 2.40598e4
    tol = args.tol
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(device)
    for i in range(args.time_steps):
        for j in range(args.Nu):
            pds[i,j] = pds[i,j] / (flp[j,0] * flp[j,1] * thick_actl)
    pds = torch.tensor(pds, dtype=torch.float64).to(device)
    
    def density_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 2.33e3, 2.65e3)

    def heat_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 751.1, 680.0)
    
    def kappa_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 100, 1.2)

    def pd_func(x, Nu2):
        x2 = x[2]
        x1 = x[1]
        x0 = x[0]
        ret = torch.zeros_like(x0)
        condition = (x2 > 5.85e-4 - thick_actl) & (x2<5.85e-4) & (x0 >= flp[Nu2][2]) & (x0 < (flp[Nu2][0] + flp[Nu2][2] +tol)) & (x1 >= flp[Nu2][3]) & (x1 < (flp[Nu2][3] +flp[Nu2][1] + tol))
        ret[condition]+=1.0
        return ret

    class BoundaryX0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], args.x, args.tol)

    class BoundaryX1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[0], 0, args.tol)

    class BoundaryY0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], 0, args.tol)

    class BoundaryY1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[1], args.y, args.tol)
    class BoundaryZ0(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], args.z, args.tol)

    class BoundaryZ1(SubDomain):
      def inside(self, x, on_boundary):
          return on_boundary and near(x[2], 0, args.tol)
        
    subdomain_data ={
        'dtype': 'size_t',
        'degree': 2,
        'boundaries': [BoundaryX0(), BoundaryX1(), BoundaryY0(), BoundaryY1(), BoundaryZ0(), BoundaryZ1()]
    }

    model = PyPOD_GP(args, device, subdomain_data)

    datapath = []
    for i in range(args.time_steps):
        datapath.append('path to i-th data point')

    C, G, P = model.train(datapath, density_expression, heat_expression, pd_func, kappa_expression, h_c)
    
    TG = pd.read_csv('path to temperature gradient matrix', sep=',', header=None)
    TG = torch.from_numpy(TG.values).to(device)
    sum_p = 0
    Ps_matrix = torch.zeros(args.time_steps, args.num_modes).double().to(device)
    for i in range(args.time_steps):
        for j in range(args.num_modes):
            Ps_matrix[i][j] = (P[j] * pds[i]).sum() + TG[j][i]

    Ps_matrix = Ps_matrix.t()

    CU = model.infer(C, G, Ps_matrix, multiple=False)
    temps = model.predict_thermal(CU, multiple=False)
    if args.save_format == 'csv':
        df = pd.DataFrame(temps.detach().cpu().numpy()) #convert to a dataframe
        df.to_csv(args.save_dir + "temsp.csv",index=False, header=False) #save to file
    elif args.save_format == 'txt':
        np.savetxt(args.save_dir + "temsp.txt", temps.detach().cpu().numpy())

if __name__ == '__main__':
    args = parser.parse_args()
    run_prediction(args)