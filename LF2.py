import numpy as np
import matplotlib.pyplot as plt
import os

# --- Préparation ---
output_dir = "images_vagues_structure"
if not os.path.exists(output_dir): os.makedirs(output_dir)

# --- Paramètres ---
L_dom = 500.0
N = 1000 # Augmenté pour plus de précision
dx = L_dom / N
x = np.linspace(-L_dom/2, L_dom/2, N + 2)

g, h0, rho = 9.81, 10.0, 1000.0
a, T = 0.5, 10.0      # Onde incidente
l_obs = 20.0          # Demi-largeur de l'obstacle [-l, l]
a_obs, T_obs = 0.2, 5.0 # Paramètres du débit imposé qi(t)

dt = 0.8 * dx / np.sqrt(g * h0)
T_max = 60.0
nt = int(T_max / dt)
steps_per_second = int(1.0 / dt)

# --- Identification de l'obstacle ---
idx_l = np.argmin(np.abs(x - (-l_obs))) # Indice à x = -l
idx_r = np.argmin(np.abs(x - l_obs))    # Indice à x = l

# --- Initialisation ---
zeta = np.zeros(N + 2)
q = np.zeros(N + 2)

def qi(t):
    return 0
    # return a_obs * np.sin(2 * np.pi * t / T_obs)

def compute_flux(z, q_val):
    h = np.maximum(h0 + z, 1e-4)
    return g * (h**2 - h0**2) / 2.0 + (q_val**2 / h)

def get_invariants(z, q_val):
    h = np.maximum(h0 + z, 1e-4)
    c = np.sqrt(g * h)
    v = q_val / h
    return 2*(c - np.sqrt(g*h0)) - v, 2*(c - np.sqrt(g*h0)) + v, v-c, v+c

# --- Boucle principale ---
for n in range(nt):
    t = n * dt
    zeta_new, q_new = np.copy(zeta), np.copy(q)
    f_flux = compute_flux(zeta, q)
    L, R, lm, lp = get_invariants(zeta, q)
    
    # 1. Points intérieurs (Fluide à gauche et à droite de l'obstacle)
    indices_fluide = list(range(1, idx_l)) + list(range(idx_r + 1, N + 1))
    for j in indices_fluide:
        zeta_new[j] = 0.5*(zeta[j+1]+zeta[j-1]) - (dt/(2*dx))*(q[j+1]-q[j-1])
        q_new[j] = 0.5*(q[j+1]+q[j-1]) - (dt/(2*dx))*(f_flux[j+1]-f_flux[j-1])

    # 2. Bord gauche du domaine (Génération) [cite: 15, 21, 24]
    zeta_new[0] = a * np.sin(2 * np.pi * t / T)
    L0 = (1 + lm[1]*dt/dx)*L[0] - lm[1]*dt/dx*L[1]
    h0_n = h0 + zeta_new[0]
    q_new[0] = h0_n * (2*(np.sqrt(g*h0_n) - np.sqrt(g*h0)) - L0)

    # 3. Bord droit du domaine (Transparent) [cite: 25, 33, 35, 37]
    RNp1 = (1 - lp[-1]*dt/dx)*R[-1] + lp[-1]*dt/dx*R[-2]
    zeta_new[-1] = ((np.sqrt(g*h0) + RNp1/4.0)**2 / g) - h0
    hNp1 = h0 + zeta_new[-1]
    q_new[-1] = 2 * hNp1 * (np.sqrt(g*hNp1) - np.sqrt(g*h0))

    # 4. Obstacle : Condition qi(t) aux limites internes
    current_qi = qi(t)
    
    # Paroi gauche de l'obstacle (x = -l) : utilise l'invariant R venant de la gauche
    R_obs_l = (1 - lp[idx_l]*dt/dx)*R[idx_l] + lp[idx_l]*dt/dx*R[idx_l-1]
    q_new[idx_l] = current_qi
    zeta_new[idx_l] = ((np.sqrt(g*h0) + (R_obs_l - current_qi/h0)/2.0)**2 / g) - h0 # Approx linéaire pour zeta

    # Paroi droite de l'obstacle (x = l) : utilise l'invariant L venant de la droite
    L_obs_r = (1 + lm[idx_r]*dt/dx)*L[idx_r] - lm[idx_r]*dt/dx*L[idx_r+1]
    q_new[idx_r] = current_qi
    zeta_new[idx_r] = ((np.sqrt(g*h0) + (L_obs_r + current_qi/h0)/2.0)**2 / g) - h0

    # Zone morte (intérieur de l'obstacle)
    zeta_new[idx_l+1:idx_r] = -h0 # On "vide" l'obstacle visuellement
    q_new[idx_l+1:idx_r] = 0

    zeta, q = zeta_new, q_new

    # Sauvegarde images
    if n % steps_per_second == 0:
        plt.figure(figsize=(10, 4))
        plt.fill_between(x, -h0, zeta, color='dodgerblue', alpha=0.7)
        plt.axvspan(-l_obs, l_obs, color='gray', label="Obstacle")
        plt.title(f"Vague-Structure : t = {int(t)}s (qi={current_qi:.2f})")
        plt.ylim(-h0, a*3)
        plt.savefig(os.path.join(output_dir, f"frame_{int(t):02d}.png"))
        plt.close()

plt.show()