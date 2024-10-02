import numpy as np
import torch
from fenics import *
from dolfin import * 
from mshr import *
from mpi4py import MPI

def get_quad_points(deg, device, dtype=torch.float64):
    if deg == 1:
        return (1 / 4, 1 / 4, 1 / 4, 1 / 4)
    elif deg == 2:
        x1 = torch.tensor([[0.585410196624969], [0.138196601125011], [0.138196601125011], [0.138196601125011]], dtype=dtype).to(device)
        x2 = torch.tensor([[0.138196601125011], [0.585410196624969], [0.138196601125011], [0.138196601125011]], dtype=dtype).to(device)
        x3 = torch.tensor([[0.138196601125011], [0.138196601125011], [0.585410196624969], [0.138196601125011]], dtype=dtype).to(device)
        x4 = torch.tensor([[0.138196601125011], [0.138196601125011], [0.138196601125011], [0.585410196624969]], dtype=dtype).to(device)
        output = (x1, x2, x3, x4)
        return output
    elif deg == 3:
        x1 = torch.tensor([[0.2500000000000000], [0.2500000000000000], [0.2500000000000000], [0.2500000000000000]], dtype=dtype).to(device)
        x2 = torch.tensor([[0.5000000000000000], [0.1666666666666666], [0.1666666666666666], [0.1666666666666666]], dtype=dtype).to(device)
        x3 = torch.tensor([[0.1666666666666666], [0.5000000000000000], [0.1666666666666666], [0.1666666666666666]], dtype=dtype).to(device)
        x4 = torch.tensor([[0.1666666666666666], [0.1666666666666666], [0.5000000000000000], [0.1666666666666666]], dtype=dtype).to(device)
        x5 = torch.tensor([[0.1666666666666666], [0.1666666666666666], [0.1666666666666666], [0.5000000000000000]], dtype=dtype).to(device)
        output = (x1, x2, x3, x4, x5)
        return output
    elif deg == 4:
        x1 = torch.tensor([[0.0000000000000000], [0.5000000000000000], [0.5000000000000000], [0.0]], dtype=dtype).to(device)
        x2 = torch.tensor([[0.5000000000000000], [0.0000000000000000], [0.5000000000000000], [0.0]], dtype=dtype).to(device)
        x3 = torch.tensor([[0.5000000000000000], [0.5000000000000000], [0.0000000000000000], [0.0]], dtype=dtype).to(device)
        x4 = torch.tensor([[0.5000000000000000], [0.0000000000000000], [0.0000000000000000], [0.5]], dtype=dtype).to(device)
        x5 = torch.tensor([[0.0000000000000000], [0.5000000000000000], [0.0000000000000000], [0.5]], dtype=dtype).to(device)
        x6 = torch.tensor([[0.0000000000000000], [0.0000000000000000], [0.5000000000000000], [0.5]], dtype=dtype).to(device)
        x7 = torch.tensor([[0.6984197043243866], [0.1005267652252045], [0.1005267652252045], [0.1005267652252045]], dtype=dtype).to(device)
        x8 = torch.tensor([[0.1005267652252045], [0.1005267652252045], [0.1005267652252045], [0.6984197043243866]], dtype=dtype).to(device)
        x9 = torch.tensor([[0.1005267652252045], [0.1005267652252045], [0.6984197043243866], [0.1005267652252045]], dtype=dtype).to(device)
        x10 = torch.tensor([[0.1005267652252045],[0.6984197043243866], [0.1005267652252045], [0.1005267652252045]], dtype=dtype).to(device)
        x11 = torch.tensor([[0.0568813795204234],[0.3143728734931922], [0.3143728734931922], [0.3143728734931922]], dtype=torch.float64).to(device)
        x12 = torch.tensor([[0.3143728734931922],[0.3143728734931922], [0.3143728734931922], [0.0568813795204234]], dtype=torch.float64).to(device)
        x13 = torch.tensor([[0.3143728734931922],[0.3143728734931922], [0.0568813795204234], [0.3143728734931922]], dtype=torch.float64).to(device)
        x14 = torch.tensor([[0.3143728734931922],[0.0568813795204234], [0.3143728734931922], [0.3143728734931922]], dtype=torch.float64).to(device)
        output = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14)
        return output

    elif deg == 5:
        x1 = torch.tensor([[0.2500000000000000], [0.2500000000000000], [0.2500000000000000], [0.2500000000000000]], dtype=torch.float64).to(device)
        x2 = torch.tensor([[0.0000000000000000], [0.3333333333333333], [0.3333333333333333], [0.3333333333333333]], dtype=torch.float64).to(device)
        x3 = torch.tensor([[0.3333333333333333], [0.3333333333333333], [0.3333333333333333], [0.0000000000000000]], dtype=torch.float64).to(device)
        x4 = torch.tensor([[0.3333333333333333], [0.3333333333333333], [0.0000000000000000], [0.3333333333333333]], dtype=torch.float64).to(device)
        x5 = torch.tensor([[0.3333333333333333], [0.0000000000000000], [0.3333333333333333], [0.3333333333333333]], dtype=torch.float64).to(device)
        x6 = torch.tensor([[0.7272727272727273], [0.0909090909090909], [0.0909090909090909], [0.0909090909090909]], dtype=torch.float64).to(device)
        x7 = torch.tensor([[0.0909090909090909], [0.7272727272727273], [0.0909090909090909], [0.0909090909090909]], dtype=torch.float64).to(device)
        x8 = torch.tensor([[0.0909090909090909], [0.0909090909090909], [0.7272727272727273], [0.0909090909090909]], dtype=torch.float64).to(device)
        x9 = torch.tensor([[0.0909090909090909], [0.0909090909090909], [0.0909090909090909], [0.7272727272727273]], dtype=torch.float64).to(device)
        x10 = torch.tensor([[0.4334498464263357], [0.0665501535736643], [0.0665501535736643], [0.4334498464263357]], dtype=torch.float64).to(device)
        x11 = torch.tensor([[0.4334498464263357], [0.4334498464263357], [0.0665501535736643], [0.0665501535736643]], dtype=torch.float64).to(device)
        x12 = torch.tensor([[0.4334498464263357], [0.0665501535736643], [0.4334498464263357], [0.0665501535736643]], dtype=torch.float64).to(device)
        x13 = torch.tensor([[0.0665501535736643], [0.4334498464263357], [0.4334498464263357], [0.0665501535736643]], dtype=torch.float64).to(device)
        x14 = torch.tensor([[0.0665501535736643], [0.4334498464263357], [0.0665501535736643], [0.4334498464263357]], dtype=torch.float64).to(device)
        x15 = torch.tensor([[0.0665501535736643], [0.0665501535736643], [0.4334498464263357], [0.4334498464263357]], dtype=torch.float64).to(device)
        output = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15)
        return output
    elif deg == 6:
        x1 = torch.tensor([[0.3561913862225449], [0.2146028712591517], [0.2146028712591517], [0.2146028712591517]], dtype=torch.float64).to(device)
        x2 = torch.tensor([[0.2146028712591517], [0.3561913862225449], [0.2146028712591517], [0.2146028712591517]], dtype=torch.float64).to(device)
        x3 = torch.tensor([[0.2146028712591517], [0.2146028712591517], [0.3561913862225449], [0.2146028712591517]], dtype=torch.float64).to(device)
        x4 = torch.tensor([[0.2146028712591517], [0.2146028712591517], [0.2146028712591517], [0.3561913862225449]], dtype=torch.float64).to(device)
        x5 = torch.tensor([[0.8779781243961660], [0.0406739585346113], [0.0406739585346113], [0.0406739585346113]], dtype=torch.float64).to(device)
        x6 = torch.tensor([[0.0406739585346113], [0.8779781243961660], [0.0406739585346113], [0.0406739585346113]], dtype=torch.float64).to(device)
        x7 = torch.tensor([[0.0406739585346113], [0.0406739585346113], [0.8779781243961660], [0.0406739585346113]], dtype=torch.float64).to(device)
        x8 = torch.tensor([[0.0406739585346113], [0.0406739585346113], [0.0406739585346113], [0.8779781243961660]], dtype=torch.float64).to(device)
        x9 = torch.tensor([[0.0329863295731731], [0.3223378901422757], [0.3223378901422757], [0.3223378901422757]], dtype=torch.float64).to(device)
        x10 = torch.tensor([[0.3223378901422757], [0.0329863295731731], [0.3223378901422757], [0.3223378901422757]], dtype=torch.float64).to(device)
        x11 = torch.tensor([[0.3223378901422757], [0.3223378901422757], [0.0329863295731731], [0.3223378901422757]], dtype=torch.float64).to(device)
        x12 = torch.tensor([[0.3223378901422757], [0.3223378901422757], [0.3223378901422757], [0.0329863295731731]], dtype=torch.float64).to(device)
        x13 = torch.tensor([[0.2696723314583159], [0.0636610018750175], [0.0636610018750175], [0.6030056647916491]], dtype=torch.float64).to(device)
        x14 = torch.tensor([[0.0636610018750175], [0.2696723314583159], [0.0636610018750175], [0.6030056647916491]], dtype=torch.float64).to(device)
        x15 = torch.tensor([[0.0636610018750175], [0.0636610018750175], [0.2696723314583159], [0.6030056647916491]], dtype=torch.float64).to(device)
        x16 = torch.tensor([[0.6030056647916491], [0.0636610018750175], [0.0636610018750175], [0.2696723314583159]], dtype=torch.float64).to(device)
        x17 = torch.tensor([[0.0636610018750175], [0.6030056647916491], [0.0636610018750175], [0.2696723314583159]], dtype=torch.float64).to(device)
        x18 = torch.tensor([[0.0636610018750175], [0.0636610018750175], [0.6030056647916491], [0.2696723314583159]], dtype=torch.float64).to(device)
        x19 = torch.tensor([[0.0636610018750175], [0.2696723314583159], [0.6030056647916491], [0.0636610018750175]], dtype=torch.float64).to(device)
        x20 = torch.tensor([[0.2696723314583159], [0.6030056647916491], [0.0636610018750175], [0.0636610018750175]], dtype=torch.float64).to(device)
        x21 = torch.tensor([[0.6030056647916491], [0.0636610018750175], [0.2696723314583159], [0.0636610018750175]], dtype=torch.float64).to(device)
        x22 = torch.tensor([[0.0636610018750175], [0.6030056647916491], [0.2696723314583159], [0.0636610018750175]], dtype=torch.float64).to(device)
        x23 = torch.tensor([[0.2696723314583159], [0.0636610018750175], [0.6030056647916491], [0.0636610018750175]], dtype=torch.float64).to(device)
        x24 = torch.tensor([[0.6030056647916491], [0.2696723314583159], [0.0636610018750175], [0.0636610018750175]], dtype=torch.float64).to(device)
        output = (x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24)
        return output
    else:
        raise NotImplementedError("Degree not implemented")

def get_weights(deg, device, dtype=torch.float64):
    if deg == 1:
        return torch.tensor(0.25, device=device, dtype=dtype)
    elif deg == 2:
        return torch.tensor([0.25, 0.25, 0.25, 0.25], device=device, dtype=dtype)
    elif deg == 3:
        w = torch.zeros(5, device=device, dtype=dtype)
        w[0] = -0.8
        w[1:5] = 0.45
        return w
    elif deg == 4:
        w = torch.zeros(14, device=device, dtype=dtype)
        w[0:6] = 0.0190476190476190
        w[6:10] = 0.0885898247429807
        w[10:14] = 0.1328387466855907
        return w
    elif deg == 5:
        w = torch.zeros(15, device=device, dtype=dtype)
        w[0] = 0.1817020685825351
        w[1:5] = 0.0361607142857143
        w[5:9] = 0.0698714945161738
        w[9:15] = 0.0656948493683187
        return 2
    elif deg == 6:
        w = torch.zeros(24, device=device, dtype=dtype)
        w[0:4] = 0.0399227502581679
        w[4:8] = 0.0100772110553207
        w[8:12] = 0.0553571815436544
        w[12:24] = 0.0482142857142857
        return w
    else:
        raise NotImplementedError("Degree not implemented")

def get_interp_coord(coord, quad_pts):
    output = []
    for w in quad_pts:
        output.append(torch.stack([(w * coord[0]).sum(dim=0), (w * coord[1]).sum(dim=0), (w * coord[2]).sum(dim=0)], dim=0))
    return output

def get_interpolation(data, degree, weights=None, dtype=torch.float64):
    if degree == 2:
        if weights:
            x1, x2, x3, x4 = weights
        else: 
            x1, x2, x3, x4 = get_quad_points(2, data.device, dtype=dtype)
        u11 = (x1 * data).sum(dim=0)
        u12 = (x2 * data).sum(dim=0)
        u13 = (x3 * data).sum(dim=0)
        u14 = (x4 * data).sum(dim=0)
        output = (u11, u12, u13, u14)
        return output
    elif degree == 3:
        if weights:
            x1, x2, x3, x4, x5 = weights
        else: 
            x1, x2, x3, x4, x5 = get_quad_points(3, data.device, dtype=dtype)
        u11 = (x1 * data).sum(dim=0)
        u12 = (x2 * data).sum(dim=0)
        u13 = (x3 * data).sum(dim=0)
        u14 = (x4 * data).sum(dim=0)
        u15 = (x5 * data).sum(dim=0)
        output = (u11, u12, u13, u14, u15)
        return output
    elif degree == 4:
        if weights:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = weights
        else:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14 = get_quad_points(4, data.device, dtype=dtype)
        u11 = (x1 * data).sum(dim=0)
        u12 = (x2 * data).sum(dim=0)
        u13 = (x3 * data).sum(dim=0)
        u14 = (x4 * data).sum(dim=0)
        u15 = (x5 * data).sum(dim=0)
        u16 = (x6 * data).sum(dim=0)
        u17 = (x7 * data).sum(dim=0)
        u18 = (x8 * data).sum(dim=0)
        u19 = (x9 * data).sum(dim=0)
        u110 = (x10 * data).sum(dim=0)
        u111 = (x11 * data).sum(dim=0)
        u112 = (x12 * data).sum(dim=0)
        u113 = (x13 * data).sum(dim=0)
        u114 = (x14 * data).sum(dim=0)
        output = (u11, u12, u13, u14, u15, u16, u17, u18, u19, u110, u111, u112, u113, u114)
        return output
    elif degree==5:
        if weights:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = weights
        else:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15 = get_quad_points(5, data.device, dtype=dtype)
        u11 = (x1 * data).sum(dim=0)
        u12 = (x2 * data).sum(dim=0)
        u13 = (x3 * data).sum(dim=0)
        u14 = (x4 * data).sum(dim=0)
        u15 = (x5 * data).sum(dim=0)
        u16 = (x6 * data).sum(dim=0)
        u17 = (x7 * data).sum(dim=0)
        u18 = (x8 * data).sum(dim=0)
        u19 = (x9 * data).sum(dim=0)
        u110 = (x10 * data).sum(dim=0)
        u111 = (x11 * data).sum(dim=0)
        u112 = (x12 * data).sum(dim=0)
        u113 = (x13 * data).sum(dim=0)
        u114 = (x14 * data).sum(dim=0)
        u115 = (x15 * data).sum(dim=0)
        output = (u11, u12, u13, u14, u15, u16, u17, u18, u19, u110, u111, u112, u113, u114, u115)
        return output
    else:
        if weights:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24 = weights
        else:
            x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13, x14, x15, x16, x17, x18, x19, x20, x21, x22, x23, x24 = get_quad_points(6, data.device, dtype=dtype)
        u11 = (x1 * data).sum(dim=0)
        u12 = (x2 * data).sum(dim=0)
        u13 = (x3 * data).sum(dim=0)
        u14 = (x4 * data).sum(dim=0)
        u15 = (x5 * data).sum(dim=0)
        u16 = (x6 * data).sum(dim=0)
        u17 = (x7 * data).sum(dim=0)
        u18 = (x8 * data).sum(dim=0)
        u19 = (x9 * data).sum(dim=0)
        u110 = (x10 * data).sum(dim=0)
        u111 = (x11 * data).sum(dim=0)
        u112 = (x12 * data).sum(dim=0)
        u113 = (x13 * data).sum(dim=0)
        u114 = (x14 * data).sum(dim=0)
        u115 = (x15 * data).sum(dim=0)
        u116 = (x16 * data).sum(dim=0)
        u117 = (x17 * data).sum(dim=0)
        u118 = (x18 * data).sum(dim=0)
        u119 = (x19 * data).sum(dim=0)
        u120 = (x20 * data).sum(dim=0)
        u121 = (x21 * data).sum(dim=0)
        u122 = (x22 * data).sum(dim=0)
        u123 = (x23 * data).sum(dim=0)
        u124 = (x24 * data).sum(dim=0)
        output = (u11, u12, u13, u14, u15, u16, u17, u18, u19, u110, u111, u112, u113, u114, u115, u116, u117, u118, u119, u120, u121, u122, u123, u124)
        return output

def get_jacobian(mesh, size, device, dtype=torch.float64):

    jacobian = []
    for i in range(size):
        cell = Cell(mesh, i)
        jacobian.append(cell.volume())
    return torch.tensor(jacobian, dtype=dtype).to(device)

def compute_jacobian(vertices):
    J = np.array([
        [vertices[1][0] - vertices[0][0], vertices[2][0] - vertices[0][0], vertices[3][0] - vertices[0][0]],
        [vertices[1][1] - vertices[0][1], vertices[2][1] - vertices[0][1], vertices[3][1] - vertices[0][1]],
        [vertices[1][2] - vertices[0][2], vertices[2][2] - vertices[0][2], vertices[3][2] - vertices[0][2]]
        ])
    j = np.abs(np.linalg.det(J))
    return j

def barycentric_gradients(coord):
    V0, V1, V2, V3 = coord[0], coord[1], coord[2], coord[3]
    n1, n2, n3, n4 = normal_vector(V1, V2, V3), normal_vector(V0, V2, V3), normal_vector(V0, V1, V3), normal_vector(V0, V1, V2)
    g1, g2, g3, g4 = n1.t() / point_to_plane_distance(n1, V0, V1), n2.t() / point_to_plane_distance(n2, V1, V3), n3.t() / point_to_plane_distance(n3, V2, V3), n4.t() / point_to_plane_distance(n4, V3, V0)
    g1[torch.logical_and(g1>=-1e-4, g1<=1e-4)] = 0.0
    g2[torch.logical_and(g2>=-1e-4, g2<=1e-4)] = 0.0
    g3[torch.logical_and(g3>=-1e-4, g3<=1e-4)] = 0.0
    g4[torch.logical_and(g4>=-1e-4, g4<=1e-4)] = 0.0
    return (g1, g2, g3, g4)

def normal_vector(a, b, c):
    x = torch.cross(b - a, c - a)
    denom = (x * x).sum(dim=-1).sqrt()
    return (x.t() / (denom)).t()

def point_to_plane_distance(n, point, vertex_on_plane):
    x = point - vertex_on_plane
    return (x * n).sum(dim=-1)

    