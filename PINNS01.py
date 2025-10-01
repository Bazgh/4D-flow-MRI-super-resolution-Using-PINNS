# -*- coding: utf-8 -*-
"""
PINNS.py — GPU-mem-safe:
- All big data (mesh, inlet/wall, sparse) lives on CPU (NumPy/CPU tensors).
- Each training step: move ONLY the current batch to GPU.
- PDE autograd & optimizer steps run on GPU.
- Data loss samples a tiny subset and slices on CPU first, then moves small slices.
- Geometry latent expanded per-batch (no N-wide latent on GPU).
- AMP (autocast + GradScaler) used to cut VRAM.
"""

import os, time, subprocess, threading
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMPORARY ONLY
from vtk.util.numpy_support import numpy_to_vtk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, RandomSampler
import pandas
from stl import mesh

import matplotlib

matplotlib.use("Agg")  # headless plotting for Slurm
import matplotlib.pyplot as plt

import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.spatial import cKDTree
import pandas as pd

# -
# ----------------------------
# Paths (relative to this file)
# ----------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
CKPT_PATH = os.path.join(ROOT, "geom_pointnet_vae_k8_N400.pt")
CKPT_PATH2 = os.path.join(ROOT, "inlet_pointnet_vae_k4_N400.pt")
MESH_FILE = os.path.join(ROOT, "vel.csv")
INLET_FILE = os.path.join(ROOT, "inlet_vel.csv")
WALL_FILE = os.path.join(ROOT, "wall.csv")
SPARSE_FILE = os.path.join(ROOT, "sample_2.vtk")
RESULT_DIR = os.path.join(ROOT, "Results")
os.makedirs(RESULT_DIR, exist_ok=True)

# ----------------------------
# Device & globals
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # perf hint

# scaling/physics
"""
X_scale = 1.0
Y_scale = 1.0
Z_scale = 1.0
U_scale = 1.0
"""
U_BC_in = 1.0
Diff = 1e-3
rho = 1.0

# training knobs (start conservative; scale up later)
Flag_BC_exact = False
Lambda_BC = 20.0
Lambda_div = 1.0
batchsize = 2  # tiny to avoid OOM
colloc_per_ep = 500  # tiny to avoid OOM
internal_size = 10
epochs = 50
result_check = 5
add_l1_loss = True
use_amp = True

# domain box (for optional exact-BC shaping)
xStart, xEnd = 0.0, 1.0
yStart, yEnd = 0.0, 1.0
zStart, zEnd = 0.0, 1.0

# LR schedule
learning_rate = 1e-3
step_epoch = 1200
decay_rate = 0.1

# ----------------------------
# Load data (CPU)
# ----------------------------
print("Loading mesh:", MESH_FILE, flush=True)
csv_mesh = pandas.read_csv(MESH_FILE)
mesh_xyz = csv_mesh.iloc[:, : 3].to_numpy().astype(np.float32)
mesh_uvw = csv_mesh.iloc[:, 3: 6].to_numpy().astype(np.float32)
x=mesh_xyz[:,0:1];y=mesh_xyz[:,1:2];z=mesh_xyz[:,2:3];
u=mesh_uvw[:,0:1];v=mesh_uvw[:,1:2];w=mesh_uvw[:,2:3];
N = x.shape[0]
print("n_points of the mesh:", N, flush=True)
x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()
X_scale=x_max - x_min
Y_scale=y_max - y_min
Z_scale=z_max - z_min
u_min,u_max=u.min(),u.max()
v_min,v_max=v.min(),v.max()
w_min,w_max=w.min(),w.max()
U_scale=u_max - u_min
V_scale=v_max - v_min
W_scale=w_max - w_min
# selecting 10 random points for data loss
df_mesh = pandas.DataFrame(csv_mesh)
data_points = df_mesh.sample(n=internal_size, replace=False, random_state=42)
data_xyz = data_points.iloc[:, : 3].to_numpy().astype(np.float32)
data_uvw = data_points.iloc[:, 3: 6].to_numpy().astype(np.float32)
x_data = data_xyz[:, 0:1];
y_data = data_xyz[:, 1:2];
z_data = data_xyz[:, 2:3]
u_data = data_uvw[:, 0:1];
v_data = data_uvw[:, 1:2];
w_data = data_uvw[:, 2:3]

print("Loading inlet:", INLET_FILE, flush=True)
csv_inlet = pandas.read_csv(INLET_FILE)
inlet_xyz = csv_inlet.iloc[:, : 3].to_numpy().astype(np.float32)
inlet_uvw = csv_inlet.iloc[:, 3: 6].to_numpy().astype(np.float32)
xb_in = inlet_xyz[:, 0:1];
yb_in = inlet_xyz[:, 1:2];
zb_in = inlet_xyz[:, 2:3]

u_inlet = inlet_uvw[:, 0:1];
v_inlet = inlet_uvw[:, 1:2];
w_inlet = inlet_uvw[:, 2:3]
Inlet_vector = np.concatenate((inlet_xyz, inlet_uvw), axis=1)
print(Inlet_vector.shape)

print("Loading wall:", WALL_FILE, flush=True)
csv_wall = pandas.read_csv(WALL_FILE)
wall_xyz = csv_wall.iloc[:, : 3].to_numpy().astype(np.float32)
wall_uvw = csv_wall.iloc[:, 3: 6].to_numpy().astype(np.float32)
xb = wall_xyz[:, 0:1];
yb = wall_xyz[:, 1:2];
zb = wall_xyz[:, 2:3]
print("n_points at wall:", xb.shape[0], flush=True)

# inlet BC

w_in_BC = u_inlet
u_in_BC = v_inlet
v_in_BC = w_inlet

# wall no-slip
u_wall_BC = np.zeros_like(xb, dtype=np.float32)
v_wall_BC = np.zeros_like(yb, dtype=np.float32)
w_wall_BC = np.zeros_like(zb, dtype=np.float32)

# ----------------------------
# Geometry encoder  & Inlet encoder (latent z)
# ----------------------------
from Geometry_encoder_training import BoundaryEncoder
from Inlet_encoder_training import InletEncoder


def load_geom_encoder_and_latent(wall_to_compress: np.ndarray, k: int) -> torch.Tensor:
    enc = BoundaryEncoder(k=k).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        enc_state = {kk.replace("enc.", "", 1): vv for kk, vv in ckpt.items() if kk.startswith("enc.")}
        enc.load_state_dict(enc_state if enc_state else ckpt, strict=True)
    enc.eval()
    with torch.no_grad():
        coords_t = torch.from_numpy(wall_to_compress).unsqueeze(0).to(device)  # [1,N,3]
        mu, lv = enc(coords_t)  # [1,k], [1,k]
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    return z.squeeze(0)  # [k]


def load_inlet_encoder_and_latent(inlet_to_compress: np.ndarray, k: int) -> torch.Tensor:
    enc = InletEncoder(k=k).to(device)
    ckpt = torch.load(CKPT_PATH2, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        enc_state = {kk.replace("enc.", "", 1): vv for kk, vv in ckpt.items() if kk.startswith("enc.")}
        enc.load_state_dict(enc_state if enc_state else ckpt, strict=True)
    enc.eval()
    with torch.no_grad():
        coords_t = torch.from_numpy(inlet_to_compress).unsqueeze(0).to(device)  # [1,N,6]
        mu, lv = enc(coords_t)  # [1,k], [1,k]
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    return z.squeeze(0)  # [k]


# Instantiate with the *checkpoint's* k
geom_latent = load_geom_encoder_and_latent(wall_xyz, k=8).to(device)  # [K1]
inlet_latent = load_inlet_encoder_and_latent(Inlet_vector, k=4).to(device)  # [K2]
K1 = geom_latent.numel()
K2 = inlet_latent.numel()
input_n = 32 + K1 + K2


# Nets
# ----------------------------
class Swish(nn.Module):
    def forward(self, x): return x * torch.sigmoid(x)


class MySquared(nn.Module):
    def forward(self, x): return torch.square(x)


class MLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, depth, last_act=None):
        super().__init__()
        layers, d = [], in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), Swish()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        if last_act is not None: layers += [last_act]
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)

#first apply MLP in Internal mesh to bring it closer to the Geometry and Inlet latent space
input_coords=3
h=96 #width
depth_main_net=8
Net_internal_mesh = lambda: MLP(input_coords, h, 32, 4, last_act=nn.ReLU())

# keep models modest to save VRAM
h_nD = 32  # BC nets width
h_n = 96  # main nets width
depth_main = 8

Net1_bc_u = lambda: MLP(input_n, h_nD, 1, 5, last_act=nn.ReLU())
Net1_bc_v = lambda: MLP(input_n, h_nD, 1, 4, last_act=nn.ReLU())
Net1_bc_w = lambda: MLP(input_n, h_nD, 1, 4, last_act=nn.ReLU())


class Net_internal_mesh(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_coords, h, 32, depth_main_net)

    def forward(self, coord_xyz):
        out = self.m(coord_xyz)
        if Flag_BC_exact:
            x, y, z = coord_xyz[:, :1], coord_xyz[:, 1:2], coord_xyz[:, 2:3]
            out = out * (x - xStart) * (y - yStart) * (y - yEnd) + (-0.9 * z + 1.0) + (y - yStart) * (y - yEnd) * (
                        z - zStart) * (z - zEnd)
        return out


class Net2_u(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)

    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart) * (y - yStart) * (y - yEnd) + (-0.9 * z + 1.0) + (y - yStart) * (y - yEnd) * (
                        z - zStart) * (z - zEnd)
        return out


class Net2_v(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)

    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd) * (z - zStart) * (z - zEnd) + (
                        -0.9 * z + 1.0)
        return out


class Net2_w(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)

    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y, z = xin[:, :1], xin[:, 1:2], xin[:, 2:3]
            out = out * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd) * (z - zStart) * (z - zEnd) + 0.0
        return out


class Net2_p(nn.Module):
    def __init__(self): super().__init__(); self.m = MLP(input_n, h_n, 1, depth_main)

    def forward(self, xin):
        out = self.m(xin)
        if Flag_BC_exact:
            x, y = xin[:, :1], xin[:, 1:2]
            out = out * (x - xStart) * (x - xEnd) * (y - yStart) * (y - yEnd) + (-0.9 * z + 1.0)
        return out


# ----------------------------
# Losses (GPU)
# ----------------------------
def criterion_pde(net_u, net_v, net_w, net_p,net_internal_mesh, x, y, z, geom_k, inlet_k):
    # x,y,z,geom_k,inlet_k are small batch tensors on GPU
    x.requires_grad_(True);
    y.requires_grad_(True);
    z.requires_grad_(True)
    net_internal_mesh_in = torch.cat((x, y, z), 1)
    mesh_latent=net_internal_mesh(net_internal_mesh_in)
    net_in = torch.cat((mesh_latent, geom_k, inlet_k), 1)

    u = net_u(net_in).view(-1, 1)
    v = net_v(net_in).view(-1, 1)
    w = net_w(net_in).view(-1, 1)
    P = net_p(net_in).view(-1, 1)

    ones_x = torch.ones_like(x);
    ones_y = torch.ones_like(y);
    ones_z = torch.ones_like(z)

    u_x = torch.autograd.grad(u, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    u_z = torch.autograd.grad(u, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    u_zz = torch.autograd.grad(u_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    v_y = torch.autograd.grad(v, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_yy = torch.autograd.grad(v_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    v_z = torch.autograd.grad(v, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    v_zz = torch.autograd.grad(v_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    w_x = torch.autograd.grad(w, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_xx = torch.autograd.grad(w_x, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    w_y = torch.autograd.grad(w, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_yy = torch.autograd.grad(w_y, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    w_z = torch.autograd.grad(w, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]
    w_zz = torch.autograd.grad(w_z, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]  # fixed

    P_x = torch.autograd.grad(P, x, grad_outputs=ones_x, create_graph=True, only_inputs=True)[0]
    P_y = torch.autograd.grad(P, y, grad_outputs=ones_y, create_graph=True, only_inputs=True)[0]
    P_z = torch.autograd.grad(P, z, grad_outputs=ones_z, create_graph=True, only_inputs=True)[0]

    XX_scale = U_scale * (X_scale ** 2)
    YY_scale = V_scale * (Y_scale ** 2)
    ZZ_scale=W_scale*(Z_scale ** 2)
    UU_scale = U_scale ** 2
    VV_scale = V_scale ** 2
    WW_scale = W_scale ** 2

    loss_x = u * u_x / X_scale + v * u_y / Y_scale + w * u_z / Z_scale \
             - Diff * (u_xx / XX_scale + u_yy / YY_scale + u_zz / ZZ_scale) \
             + (1 / rho) * (P_x / (X_scale * UU_scale))
    loss_y = u * v_x / X_scale + v * v_y / Y_scale + w * v_z / Z_scale \
             - Diff * (v_xx / XX_scale + v_yy / YY_scale + v_zz / ZZ_scale) \
             + (1 / rho) * (P_y / (Y_scale * VV_scale))
    loss_z = u * w_x / X_scale + v * w_y / Y_scale + w * w_z / Z_scale \
             - Diff * (w_xx / XX_scale + w_yy / YY_scale + w_zz / ZZ_scale) \
             + (1 / rho) * (P_z / (Z_scale * WW_scale))
    # Continuity equation with scaling
    loss_c = (u_x / X_scale) + (v_y / Y_scale) + (w_z / Z_scale)

    mse = nn.MSELoss()

    mse_loss =  (mse(loss_x, torch.zeros_like(loss_x)) +
            mse(loss_y, torch.zeros_like(loss_y)) +
            mse(loss_z, torch.zeros_like(loss_z)) +
            mse(loss_c, torch.zeros_like(loss_c)) * Lambda_div)
    loss = mse_loss
    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(loss_x, torch.zeros_like(loss_x)) +
                   l1(loss_y, torch.zeros_like(loss_y)) +
                   l1(loss_z, torch.zeros_like(loss_z)) +
                   l1(loss_c, torch.zeros_like(loss_c)) * Lambda_div)
        loss = mse_loss + l1_loss
    return loss

def loss_bc(net_u, net_v, net_w,Net_internal_mesh,
            xb, yb, zb, ub, vb, wb,
            xb_in, yb_in, zb_in, ub_in, vb_in, wb_in,
            geom_wall, geom_in, Inlet_wall, Inlet_in):
    nin_wall_ = torch.cat((xb, yb, zb), 1)
    mesh_latent_ = Net_internal_mesh(nin_wall_)
    nin_wall = torch.cat((mesh_latent_, geom_wall, Inlet_wall), 1)
    uw = net_u(nin_wall);
    vw = net_v(nin_wall);
    ww = net_w(nin_wall)
    nin_in_ = torch.cat((xb_in, yb_in, zb_in), 1)
    mesh_latent_in = Net_internal_mesh(nin_in_ )
    nin_in = torch.cat((mesh_latent_in , geom_in, Inlet_in), 1)
    ui = net_u(nin_in);
    vi = net_v(nin_in);
    wi = net_w(nin_in)
    mse = nn.MSELoss()

    loss = (mse(uw, ub) + mse(vw, vb) + mse(ww, wb) +
            mse(ui, ub_in) + mse(vi, vb_in) + mse(wi, wb_in))
    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(uw, ub) + l1(vw, vb) + l1(ww, wb) +
                   l1(ui, ub_in) + l1(vi, vb_in) + l1(wi, wb_in)
                   )
        loss = loss + l1_loss
    return loss

def loss_data(net_u, net_v, net_w,Net_internal_mesh,
              x_data, y_data, z_data, u_data, v_data, w_data,
              geom_k, inlet_k
              ):
    nin_data_ = torch.cat((x_data, y_data, z_data), 1)
    data_latent = Net_internal_mesh(nin_data_)
    nin_data = torch.cat((data_latent, geom_k, inlet_k), 1)
    u_prediction = net_u(nin_data);
    v_prediction = net_v(nin_data);
    w_prediction = net_w(nin_data);
    mse = nn.MSELoss()

    loss = (mse(u_prediction, u_data) + mse(v_prediction, v_data) + mse(w_prediction, w_data))

    if add_l1_loss:
        l1 = nn.L1Loss()
        l1_loss = (l1(u_prediction, u_data) + l1(v_prediction, v_data) + l1(w_prediction, w_data))
        loss = loss + l1_loss

    return loss

# ----------------------------
# Training (CPU dataset; GPU compute)
# ----------------------------
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler(enabled=use_amp)


def train():
    # CPU tensors for dataset
    x_cpu = torch.from_numpy(x).float()
    y_cpu = torch.from_numpy(y).float()
    z_cpu = torch.from_numpy(z).float()
    dataset = TensorDataset(x_cpu, y_cpu, z_cpu)
    sampler = RandomSampler(dataset, replacement=True, num_samples=colloc_per_ep)
    loader = DataLoader(dataset, batch_size=batchsize, sampler=sampler,
                        num_workers=0, pin_memory=True, drop_last=True)

    # small BC/sparse tensors — prepare GPU copies ONCE
    xb_t = torch.from_numpy(xb).float().to(device)
    yb_t = torch.from_numpy(yb).float().to(device)
    zb_t = torch.from_numpy(zb).float().to(device)
    ub_t = torch.from_numpy(u_wall_BC).float().to(device)
    vb_t = torch.from_numpy(v_wall_BC).float().to(device)
    wb_t = torch.from_numpy(w_wall_BC).float().to(device)

    xbi_t = torch.from_numpy(xb_in).float().to(device)
    ybi_t = torch.from_numpy(yb_in).float().to(device)
    zbi_t = torch.from_numpy(zb_in).float().to(device)
    ubi_t = torch.from_numpy(u_in_BC).float().to(device)
    vbi_t = torch.from_numpy(v_in_BC).float().to(device)
    wbi_t = torch.from_numpy(w_in_BC).float().to(device)

    xdata_t = torch.from_numpy(x_data).float().to(device)
    ydata_t = torch.from_numpy(y_data).float().to(device)
    zdata_t = torch.from_numpy(z_data).float().to(device)
    udata_t = torch.from_numpy(u_data).float().to(device)
    vdata_t = torch.from_numpy(v_data).float().to(device)
    wdata_t = torch.from_numpy(w_data).float().to(device)

    # nets & optims
    net_u, net_v, net_w, net_p,net_internal_mesh = Net2_u().to(device), Net2_v().to(device), Net2_w().to(device), Net2_p().to(device),Net_internal_mesh().to(device)

    def init_kaiming(m):
        if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight)

    for net in (net_u, net_v, net_w, net_p): net.apply(init_kaiming)

    opt_u = optim.Adam(net_u.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    opt_v = optim.Adam(net_v.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    opt_w = optim.Adam(net_w.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)
    opt_p = optim.Adam(net_p.parameters(), lr=learning_rate, betas=(0.9, 0.99), eps=1e-15)

    sch_u = torch.optim.lr_scheduler.StepLR(opt_u, step_size=step_epoch, gamma=decay_rate)
    sch_v = torch.optim.lr_scheduler.StepLR(opt_v, step_size=step_epoch, gamma=decay_rate)
    sch_w = torch.optim.lr_scheduler.StepLR(opt_w, step_size=step_epoch, gamma=decay_rate)
    sch_p = torch.optim.lr_scheduler.StepLR(opt_p, step_size=step_epoch, gamma=decay_rate)

    t0 = time.time()
    for ep in range(1, epochs + 1):
        net_u.train()
        net_v.train()
        net_w.train()
        net_p.train()

        epoch_time = time.time()
        eq_tot = bc_tot = dat_tot = 0.0;
        nb = 0

        for (x_in_cpu, y_in_cpu, z_in_cpu) in loader:
            # move ONLY the batch to GPU
            x_in = x_in_cpu.to(device, non_blocking=True)
            y_in = y_in_cpu.to(device, non_blocking=True)
            z_in = z_in_cpu.to(device, non_blocking=True)
            geomN = geom_latent.view(1, -1).to(device).expand(x_in.shape[0], -1)
            inletN = inlet_latent.view(1, -1).to(device).expand(x_in.shape[0], -1)

            opt_u.zero_grad();
            opt_v.zero_grad();
            opt_w.zero_grad();
            opt_p.zero_grad()

            with autocast(enabled=use_amp):
                # PDE on collocation batch
                leq = criterion_pde(net_u, net_v, net_w, net_p,net_internal_mesh, x_in.float(), y_in.float(), z_in.float(), geomN.float(),
                                    inletN.float())

                # BC on small sets (GPU)
                gW = geom_latent.view(1, -1).to(device).expand(xb_t.size(0), -1)
                gI = geom_latent.view(1, -1).to(device).expand(xbi_t.size(0), -1)
                inW = inlet_latent.view(1, -1).to(device).expand(xb_t.size(0), -1)
                inI = inlet_latent.view(1, -1).to(device).expand(xbi_t.size(0), -1)
                lbc = loss_bc(net_u, net_v, net_w,net_internal_mesh,
                              xb_t, yb_t, zb_t, ub_t, vb_t, wb_t,
                              xbi_t, ybi_t, zbi_t, ubi_t, vbi_t, wbi_t,
                              gW, gI, inW, inI)

                gdata = geom_latent.view(1, -1).to(device).expand(xdata_t.size(0), -1)
                indata = inlet_latent.view(1, -1).to(device).expand(xdata_t.size(0), -1)
                ldata = loss_data(net_u, net_v, net_w,net_internal_mesh, xdata_t, ydata_t, zdata_t, udata_t, vdata_t, wdata_t, gdata,
                                  indata)

        loss = leq + Lambda_BC * lbc + ldata
        scaler.scale(loss).backward()
        scaler.step(opt_u);
        scaler.step(opt_v);
        scaler.step(opt_w);
        scaler.step(opt_p)
        scaler.update()

        eq_tot += leq.item();
        bc_tot += lbc.item();
        dat_tot += ldata.item();
        nb += 1

        sch_u.step();
        sch_v.step();
        sch_w.step();
        sch_p.step()
        print(f"Epoch {ep:04d} | Loss eqn {eq_tot / nb:.3e}  Loss BC {bc_tot / nb:.3e}  "
              f"Loss data {dat_tot / nb:.3e}  lr {opt_u.param_groups[0]['lr']:.2e}, epoch time: {time.time() - epoch_time:.2f}s ",
              flush=True)

        if ep > 0 and (ep % result_check == 0):
            print("Starting the evaluation")
            net_u.eval();
            net_v.eval();
            net_w.eval()

            # CHUNKED, NO-GRAD INFERENCE
            N = mesh_xyz.shape[0]
            u_pred = np.empty((N, 1), dtype=np.float32)
            v_pred = np.empty((N, 1), dtype=np.float32)
            w_pred = np.empty((N, 1), dtype=np.float32)

            B = 50000  # tune if needed
            g1 = geom_latent.view(1, -1).to(device)
            i1 = inlet_latent.view(1, -1).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
                for s in range(0, N, B):
                    e = min(s + B, N)
                    xi = torch.from_numpy(x[s:e]).float().to(device)  # x,y,z are numpy arrays of shape (N,1)
                    yi = torch.from_numpy(y[s:e]).float().to(device)
                    zi = torch.from_numpy(z[s:e]).float().to(device)
                    coords_i = torch.cat((xb, yb, zb), 1)
                    coord_latent_ = Net_internal_mesh(coords_i)

                    gi = g1.expand(e - s, -1)
                    ii = i1.expand(e - s, -1)
                    nin = torch.cat((coord_latent_, gi, ii), dim=1)
                    u_pred[s:e] = net_u(nin).float().cpu().numpy()
                    v_pred[s:e] = net_v(nin).float().cpu().numpy()
                    w_pred[s:e] = net_w(nin).float().cpu().numpy()

            # QUICK SCATTER PNGs
            def save_scatter(vals, name):
                fig = plt.figure(figsize=(6, 5))
                ax = fig.add_subplot(111, projection='3d')
                sc = ax.scatter(x[:, 0], y[:, 0], z[:, 0], s=1, c=vals[:, 0])
                fig.colorbar(sc);
                ax.set_title(name);
                fig.tight_layout()
                fig.savefig(os.path.join(RESULT_DIR, f"{name}.png"), dpi=150);
                plt.close(fig)

            save_scatter(u_pred, f"u_pred_{ep}")
            save_scatter(v_pred, f"v_pred_{ep}")
            save_scatter(w_pred, f"w_pred_{ep}")
            print("Saved figures in", RESULT_DIR, flush=True)

            # WRITE A VTK FILE FROM CSV POINTS (PolyData .vtp)
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(N)
            for i, (xx, yy, zz) in enumerate(mesh_xyz):  # use ORIGINAL coords
                points.SetPoint(i, float(xx), float(yy), float(zz))

            verts = vtk.vtkCellArray()
            verts.Allocate(N)
            for i in range(N):
                verts.InsertNextCell(1);
                verts.InsertCellPoint(i)

            poly = vtk.vtkPolyData()
            poly.SetPoints(points);
            poly.SetVerts(verts)

            vel = np.hstack([u_pred, v_pred, w_pred]).astype(np.float32)
            vel_vtk = numpy_to_vtk(vel, deep=True);
            vel_vtk.SetNumberOfComponents(3);
            vel_vtk.SetName("flow")
            u_vtk = numpy_to_vtk(u_pred.astype(np.float32), deep=True);
            u_vtk.SetName("u")
            v_vtk = numpy_to_vtk(v_pred.astype(np.float32), deep=True);
            v_vtk.SetName("v")
            w_vtk = numpy_to_vtk(w_pred.astype(np.float32), deep=True);
            w_vtk.SetName("w")

            pd = poly.GetPointData()
            pd.AddArray(vel_vtk);
            pd.SetActiveVectors("flow")
            pd.AddArray(u_vtk);
            pd.AddArray(v_vtk);
            pd.AddArray(w_vtk)

            out_vtp = os.path.join(RESULT_DIR, f"pinns_result_{ep}.vtp")
            w = vtk.vtkXMLPolyDataWriter()
            w.SetFileName(out_vtp);
            w.SetInputData(poly);
            w.Write()
            print("Saved:", out_vtp)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # tip: in your sbatch add:
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    train()
