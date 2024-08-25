import torch
from fenics import *
from dolfin import * 
from mshr import *
import numpy as np
import utils


class PyPOD_GP:
    def __init__(self, args, device, subdomain_config=None):
        self.num_modes = args.num_modes
        self.data_size = np.prod((args.x_dim - 1) * (args.y_dim - 1) * (args.z_dim - 1)) * 6
        self.mesh = BoxMesh(Point(0, 0, 0), Point(args.x, args.y, args.z), args.x_dim - 1, args.y_dim - 1, args.z_dim - 1) #define mesh
        self.tol = 1e-14
        self.time_steps = args.time_steps
        self.device = device
        self.Nu = args.Nu
        self.surfaces = args.surfaces
        self.subdomain_config = subdomain_config
        self.surfaces = args.surfaces
        self.steps = args.steps
        # self.final_time = args.final_time
        self.modes_data = 0
        self.degree = args.degree

    def prcoess_data(self, datapath):
        assert(self.time_steps > 0)
        #form mesh
        V = FunctionSpace(self.mesh, 'P', 1)
        #process subdomain
        if self.subdomain_config:
            subdomains = []
            for boundary in self.subdomain_config['boundaries']:
                subdomains.append(boundary)
            boundary_markers = MeshFunction(self.subdomain_config['dtype'], self.mesh, self.subdomain_config['degree'])
            boundary_markers.set_all(9999)
            subdomain_id = 0
            for subdomain in subdomains:
                subdomain.mark(boundary_markers, subdomain_id)
                subdomain_id += 1
        dummy = Constant(0)
        bc = DirichletBC(V, dummy, boundary_markers, self.surfaces)
        ds_dofs = list(bc.get_boundary_values().keys())

        u = Function(V)
        u0 = Constant(0)
        dof_coordinates = V.tabulate_dof_coordinates()
        dofmap = V.dofmap()
        cell_to_vertex = []
        coord = []
        jacobian = []
        ds_coord = []
        for i in range(self.data_size):
            dofs = dofmap.cell_dofs(i)
            cell_to_vertex.extend(dofs)
            cell = Cell(self.mesh, i)
            coord.append(dof_coordinates[dofs])
            jacobian.append(cell.volume())
        for i in range(len(ds_dofs)):
            ds_coord.append(dof_coordinates[ds_dofs[i]])

        jacobian = torch.tensor(jacobian, dtype=torch.float64).to(self.device)
        coord = torch.from_numpy(np.array(coord)).to(self.device)

        ds_coord = torch.from_numpy(np.array(ds_coord)).double().to(self.device)
        solution = [0] * self.time_steps #read in data
        vertex_data = [] #this is the value on each vertex
        cell_values = torch.zeros(self.time_steps, 4, self.data_size, dtype=torch.float64).to(self.device)

        for i in range(self.time_steps):
            filename = datapath[i]
            solution_file = HDF5File(self.mesh.mpi_comm(), filename, "r")
            solution_file.read(u, "solution")
            solution[i] = interpolate(u0,V)
            solution[i].assign(u)
            s = solution[i].vector()[:]
            vertex_data.append(s)
            data = torch.from_numpy(np.array([s[x] for x in cell_to_vertex])).to(self.device).to(cell_values.dtype)
            cell_values[i] = data.reshape(-1, 4).t() #each row is vertex i of tetrahedron
        vertex_data = torch.from_numpy(np.array(vertex_data)).to(self.device)
        return cell_values, vertex_data, cell_to_vertex, coord, jacobian, ds_dofs, ds_coord

    def train(self, datapath, density_expression, heat_expression, pd_func, kappa_expression, h_c=1.0, idx=0):
        #set the number of modes to use
        if isinstance(self.num_modes, list):
            num_modes = self.num_modes[idx]
        else:
            num_modes = self.num_modes
        #process the input
        data, vertex_data, cell_to_vertex, coord, jacobian, ds_dofs, ds_coord = self.prcoess_data(datapath)

        #manipulate coordinates for efficient computation 
        grad_coord = coord.permute(1, 0, 2).reshape(-1, 4, 3).reshape(4, -1, 3)
        temp_coord = coord.permute(2, 1, 0).reshape(3, 4, -1)
        w = utils.get_quad_points(4, self.device)
        # interp_coord = utils.get_interp_coord(temp_coord, w)
        coord = coord.permute(2, 0, 1).reshape(4, -1, 3).reshape(3, -1)

        #Compute A matrix
        A = utils.calc_A(data, self.time_steps, jacobian.to(data.dtype), self.device, self.degree)

        # #Compute POD modes
        modes_data = utils.get_modes(A, vertex_data, num_modes, self.data_size, self.time_steps, jacobian, cell_to_vertex)
    
        #read in modes
        modes = torch.zeros(num_modes, 4, self.data_size).double().to(self.device)
        ds_modes = torch.zeros(num_modes, 3, len(ds_dofs) // 3).double().to(self.device)
        for i in range(num_modes):
            m = torch.tensor([modes_data[i][x] for x in cell_to_vertex], dtype=torch.float64).to(self.device).reshape(-1, 4).t()
            d = torch.tensor([modes_data[i][x] for x in ds_dofs], dtype=torch.float64).to(self.device).reshape(-1, 3).t()
            modes[i] = m
            ds_modes[i] = d

        # #Compute C matrix
        C = utils.calc_C(modes, num_modes, jacobian, density_expression, heat_expression, coord, self.device, self.degree, torch.float64)

        #compute jacobian for surface integral
        temp_ds_coord = ds_coord.reshape(-1, 3, 3).permute(1, 0, 2).reshape(-1, 3, 3).reshape(3, -1, 3)
        t1 = temp_ds_coord[1] - temp_ds_coord[0]
        t2 = temp_ds_coord[2] - temp_ds_coord[0]
        t3 = torch.cross(t1, t2)
        ds_area = (t3 * t3).sum(dim=-1).sqrt() / 2
        
        #Compute G matrix
        G = utils.calc_G(modes, num_modes, kappa_expression, h_c, jacobian, coord, grad_coord, ds_area, ds_modes)
        
        # #Compute P vector
        P = utils.calc_P(modes, num_modes, jacobian, pd_func, coord, self.degree, self.device, torch.float64, idx)
            
        #save modes for temperature
        self.modes_data = modes_data
        return C, G, P

    def predict_thermal(self, C, G, P, multiple=True):
        #run ODE solver
        ode_solver = utils.POD_ODE_Solver(C, G, P, self.time_steps, self.num_modes, self.steps, self.sampling_interval, multiple)
        CU = ode_solver.solve()
        #predict thermal
        temps = []
        for i in range(self.time_steps):
            x = CU[i].reshape(1,-1).t()
            x = x.reshape(-1,1)
            temps.append(self.modes_data * x)
        return temps
        
        
