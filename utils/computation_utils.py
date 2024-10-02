import torch
from fenics import *
from dolfin import * 
from mshr import *
from utils.integration_utils import get_quad_points, get_interpolation, barycentric_gradients, get_weights

def calc_A(data, time_steps, jacobian, device, degree):
    A = torch.zeros((time_steps, time_steps), dtype=data.dtype).to(device)
    quad_points = get_quad_points(degree, device, A.dtype) # get barycentric weights for quadrature
    weights = get_weights(degree, device, A.dtype).unsqueeze(dim=-1)
    for i in range(time_steps):
        #interpolate data
        u1 = torch.stack(get_interpolation(data[i], degree, quad_points), dim=0)
        A[i][i] = (((u1 * u1 * weights).sum(dim=0) * jacobian).sum(dim=-1)) 
        for j in range(i + 1, time_steps):
            #interpolate data
            u2 = torch.stack(get_interpolation(data[j], degree, quad_points), dim=0)
            A[i][j] = (((u1 * u2 * weights).sum(dim=0) * jacobian).sum(dim=-1)) 
            A[j][i] = A[i][j]
    return A / (time_steps)

def get_modes(A, data, num_modes, time_steps, cell_size, cell_to_vertex, degree, dtype):
    '''
    Get the POD modes from A matrix    
    '''
    A_double = A.double()
    L, Q = torch.linalg.eigh(A_double)
    L = (L).real
    eigen_val, idx = L.topk(num_modes)
    Q = Q.t()
    denom = eigen_val * time_steps
    modes = []
    j = 0
    for i in idx:
        q = Q[i].reshape(1,-1).t()
        q = q.reshape(-1,1)
        eta_i = (q * data).sum(dim = 0) / denom[[j]]
        #set up data for integration
        v = eta_i.detach().cpu().numpy()
        solutions = torch.tensor([v[x] for x in cell_to_vertex], dtype=dtype).to(A.device)
        solutions = solutions.reshape(-1, 4).t().double()

        #get interpolations for integration
        u1 = torch.stack(get_interpolation(solutions, degree), dim=0)
        weights = get_weights(degree, device=A.device, dtype=dtype).unsqueeze(dim=-1)
        normalize_denom = ((u1 * u1 * weights).sum(dim=0) * cell_size).sum()
        eta_i = eta_i / (normalize_denom).sqrt()
        v = eta_i.cpu().detach().numpy()
        modes.append(v)
        j+=1
    return modes

def calc_C(modes, num_modes, jacobian, density_expression, heat_expression, coord, device, degree, dtype):
    C = torch.zeros(num_modes, num_modes, dtype=dtype, device=device)
    ds_coefficients = density_expression(coord) * heat_expression(coord)
    #weights for integration
    quad_points = get_quad_points(degree, device, dtype)
    weights = get_weights(degree, device=device, dtype=dtype).unsqueeze(dim=-1)
    for i in range(num_modes):
        # interpolate data
        u1 = torch.stack(get_interpolation(modes[i], 2, quad_points, dtype), dim=0)
        C[i][i] = (ds_coefficients * (u1 * u1 * weights).sum(dim=0) * jacobian).sum()
        for j in range(i + 1, num_modes):
            #interpolate data
            u2 = torch.stack(get_interpolation(modes[j], 2, quad_points, dtype), dim=0)
            C[i][j] = (ds_coefficients * (u1 * u2 * weights).sum(dim=0) * jacobian).sum()
            C[j][i] = C[i][j]
    return C

def calc_P(modes, num_modes, jacobian, pd_func, interp_coord, degree, device, dtype, Nu_idx=-1):
    quad_points = get_quad_points(degree, device, dtype) # get barycentric weights for quadrature
    weights = get_weights(degree, device=device, dtype=dtype).unsqueeze(dim=-1)
    P = torch.zeros(num_modes, 1).to(modes.device).to(dtype)
    l = []
    for i in range(len(quad_points)):
        l.append(pd_func(interp_coord[i], Nu_idx))
    power_density = torch.stack(l, dim=0)
    power_density = weights * power_density
    for i in range(num_modes):
        u1 = torch.stack(get_interpolation(modes[i], degree, quad_points), dim=0)
        p_val = ((power_density * u1).sum(dim=0) * jacobian).sum()
        P[i] = p_val
    return P

def calc_G(modes, num_modes, kappa_expression, h_c, jacobian, coord, grad_coord, ds_area, ds_modes):
    kappa = kappa_expression(coord)
    #degree 2, could be changed to be more generic 
    g1, g2, g3, g4 = barycentric_gradients(grad_coord)
    weights = torch.tensor([[[1.0/6.0], [1.0/6.0], [2.0 / 3.0]],
                   [[1.0/6.0], [2.0/3.0], [1.0 / 6.0]],
                   [[2.0/3.0], [1.0/6.0], [1.0 / 6.0]]], dtype=torch.float64).to(modes.device)
    G = torch.zeros(num_modes, num_modes).double().to(modes.device)
    for i in range(num_modes):
        d1 = torch.stack((modes[i][0], modes[i][0], modes[i][0]), dim=0)
        d2 = torch.stack((modes[i][1], modes[i][1], modes[i][1]), dim=0)
        d3 = torch.stack((modes[i][2], modes[i][2], modes[i][2]), dim=0)
        d4 = torch.stack((modes[i][3], modes[i][3], modes[i][3]), dim=0)
        x1 = g1 * d1 + g2 * d2 + g3 * d3 + g4 * d4
        u1 = weights[0] * ds_modes[i]
        u2 = weights[1] * ds_modes[i]
        u3 = weights[2] * ds_modes[i]
        G[i][i] = (kappa * x1 * x1 * jacobian).sum() + h_c * ((u1 * u1 + u2 * u2 + u3 * u3) / 3 * ds_area).sum()
        for j in range(i + 1, num_modes):
            e1 = torch.stack((modes[j][0], modes[j][0], modes[j][0]), dim=0)
            e2 = torch.stack((modes[j][1], modes[j][1], modes[j][1]), dim=0)
            e3 = torch.stack((modes[j][2], modes[j][2], modes[j][2]), dim=0)
            e4 = torch.stack((modes[j][3], modes[j][3], modes[j][3]), dim=0)
            x2 = g1 * e1 + g2 * e2 + g3 * e3 + g4 * e4
            v1 = weights[0] * ds_modes[j]
            v2 = weights[1] * ds_modes[j]
            v3 = weights[2] * ds_modes[j]
            G[i][j] =  (kappa * x1 * x2 * jacobian).sum() +  h_c * ((u1 * v1 + u2 * v2 + u3 * v3) / 4 * ds_area).sum()
            G[j][i] = G[i][j]
    return G
