import torch
from fenics import *
from dolfin import * 
from mshr import *
import numpy as np
from utils import *
from config import parser
from pypod_gp import PyPOD_GP
from mpi4py import MPI
import pandas as pd

'''
Toy example of running PyPOD-GP with some hard coded values
'''

def run_prediction(args):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    flp = [0] * args.Nu
    for i in range(args.Nu):
        flp[i] = np.loadtxt('i-th floor plan path')
    pds = [0] * args.Nu
    for i in range(args.Nu):
        pds = np.loadtxt('i-th power density path')
    thick_actl =0.00015
    h_c = 2.40598e4
    device = 'cuda:' + str(args.cuda) if int(args.cuda) >= 0 else 'cpu'
    print(device)
    for j in range(args.Nu):
        for i in range(args.time_steps):
            pds[j][i] = pds[j][i] / (flp[j][0] * flp[j][1] * thick_actl)
        pds[j] = torch.tensor(pds[j], dtype=torch.float64).to(device)

    def density_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 2.33e3, 2.65e3)

    def heat_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 751.1, 680.0)
    
    def kappa_expression(coord):
        coord = coord[2].reshape(-1, 4).t().sum(dim=0)/4
        return torch.where(coord <= 5.85e-4 + 1E-14, 100, 1.2)

    def pd_func(coord, Nu2):
        x0 = coord[0]
        x1 = coord[1]
        x2 = coord[2]
        return torch.where((x0 >= flp[Nu2][2]) & (x0 < flp[Nu2][0] + flp[Nu2][2] + 1e-14) & (x1 >= flp[Nu2][3]) & (x1 < flp[Nu2][3] + flp[Nu2][1] + 1e-14) & (x2 >= 5.85e-4 - thick_actl) & (x2 < 5.85e-4), 1.0, 0.0)

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
        'degree': args.degree,
        'boundaries': [BoundaryX0(), BoundaryX1(), BoundaryY0(), BoundaryY1(), BoundaryZ0(), BoundaryZ1()]
    }

    model = PyPOD_GP(args, pds, device, subdomain_data)

    datapath = [[0]*args.timesteps] * args.Nu
    for j in range(args.Nu):
        for i in range(args.time_steps):
            datapath[j][i] = 'training data path for j-th functional unit and i-th time step'

    C_list = []
    G_list = []
    P_list = []
    for i in range(args.Nu):
        C, G, P = model.train(datapath[i], density_expression, heat_expression, pd_func, kappa_expression, h_c, idx=i)
        C_list.append(C)
        G_list.append(G)

        Ps_matrix = torch.zeros(args.time_steps, args.num_modes).double().to(device)
        for j in range(args.time_steps):
            for k in range(args.num_modes):
                Ps_matrix[j][k] = (P[j] * pds[i][j]).sum()


        Ps_matrix = Ps_matrix.t()
        P_list.append(Ps_matrix)
    
    temps = model.predict_thermal(C_list, G_list, P_list)
    df = pd.DataFrame(temps.detach().cpu().numpy()) #convert to a dataframe
    df.to_csv("results.csv",index=False, header=False) #save to file

if __name__ == '__main__':
    args = parser.parse_args()
    run_prediction(args)