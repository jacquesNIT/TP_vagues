import numpy as np
import matplotlib.pyplot as plt
import os

# Stockage des images
output_dir = "images_vagues"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variables de la simulation

L_dom = 500.0
N = 500
dx = L_dom / N
x = np.linspace(0, L_dom, N + 2)

g = 9.81
h0 = 10.0          
rho = 1000.0      
a = 0.5          
T = 10.0           

dt = 0.9 * dx / np.sqrt(g * h0) 
T_max = 60.0       
nt = int(T_max / dt)

n_period = int(T / dt)

zeta = np.zeros(N + 2)
q = np.zeros(N + 2)
J_sum = np.zeros(N + 2) 

steps_per_second = int(1.0 / dt)


# Fonctions utiles

def compute_flux(z, q_val):
    h = h0 + z 
    h = np.maximum(h, 1e-4)
    flux_f = g * (h**2 - h0**2) / 2.0 + (q_val**2 / h) 
    return flux_f

def get_Riemann_invariants(z, q_val):
    h = h0 + z 
    h = np.maximum(h, 1e-4)
    c = np.sqrt(g * h)
    v = q_val / h
    l_minus = v - c
    l_plus = v + c
    L_inv = 2 * (c - np.sqrt(g * h0)) - v 
    R_inv = 2 * (c - np.sqrt(g * h0)) + v 
    return L_inv, R_inv, l_minus, l_plus


# Boucle principale

for n in range(nt):
    t = n * dt
    zeta_new = np.copy(zeta)
    q_new = np.copy(q)
    
    f_flux = compute_flux(zeta, q)
    L_old, R_old, l_minus, l_plus = get_Riemann_invariants(zeta, q)
    
    # Points intérieurs 
    zeta_new[1:-1] = 0.5 * (zeta[2:] + zeta[:-2]) - (dt / (2 * dx)) * (q[2:] - q[:-2])
    q_new[1:-1] = 0.5 * (q[2:] + q[:-2]) - (dt / (2 * dx)) * (f_flux[2:] - f_flux[:-2])
    
    # Bord gauche 
    zeta_new[0] = a * np.sin(2 * np.pi * t / T)
    h0_n = h0 + zeta_new[0]
    L0_new = (1 + l_minus[1] * dt / dx) * L_old[0] - (l_minus[1] * dt / dx) * L_old[1]
    q_new[0] = h0_n * (2 * (np.sqrt(g * h0_n) - np.sqrt(g * h0)) - L0_new)
    
    # Bord droit 
    RNp1_new = (1 - l_plus[-1] * dt / dx) * R_old[-1] + (l_plus[-1] * dt / dx) * R_old[-2]
    sqrt_gh_Np1 = np.sqrt(g * h0) + RNp1_new / 4.0
    zeta_new[-1] = (sqrt_gh_Np1**2 / g) - h0
    hNp1_new = h0 + zeta_new[-1]
    q_new[-1] = 2 * hNp1_new * (np.sqrt(g * hNp1_new) - np.sqrt(g * h0))
    
    zeta, q = zeta_new, q_new

    # Calcul de J
    if n >= (nt - n_period):
        h = h0 + zeta
        h = np.maximum(h, 1e-4)
        J_instant = rho * q * (g * zeta + 0.5 * (q**2 / h**2))
        J_sum += J_instant

    # Sauvegarde des images
    if n % steps_per_second == 0:
        plt.figure(figsize=(10, 4))
        plt.plot(x, zeta, color='dodgerblue', lw=2)
        plt.title(f"Avancée de la vague - Temps : {int(t)}s")
        plt.xlabel("Position x (m)")
        plt.ylabel("Élévation $\zeta$ (m)")
        plt.ylim(-a*1.5, a*1.5) 
        plt.grid(True, alpha=0.3)
        
        filename = os.path.join(output_dir, f"frame_{int(t):02d}.png")
        plt.savefig(filename)
        plt.close() 

J_mean = J_sum / n_period


# Visualisation 
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(x, zeta, color='dodgerblue', lw=2)
ax1.set_title(f"Élévation $\zeta$ à t = {T_max}s")
ax1.set_ylabel("$\zeta$ (m)")
ax1.grid(True, alpha=0.3)

ax2.plot(x, J_mean, color='red', lw=2)
ax2.set_title(f"Flux d'énergie moyen $\langle J \\rangle$ sur une période")
ax2.set_xlabel("Position x (m)")
ax2.set_ylabel("J (W/m)")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()