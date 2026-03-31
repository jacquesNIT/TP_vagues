import numpy as np 
import matplotlib.pyplot as plt 
import math as m


## Partie 1 
N = 1024
L = 100.0 

dx = L / N 
x = np.arange(N) * dx 

# Fonction test
ft = np.exp(-0.25 * (x - L/2)**2)

## Partie 2 
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

## Partie 3 
# Dérivée spectrale
f = np.fft.fft(ft)
df = 1j * k * f
df_spec = np.fft.ifft(df).real 

# Dérivée Analytique
df1 = -0.5 * (x - L/2) * ft

# Comparaison
plt.figure(figsize=(10,5))
plt.plot(x, df1, label="Analytique", lw=2, alpha=0.6)
plt.plot(x, df_spec, '--', label="Spectrale", lw=2)
plt.title("Comparaison de la dérivée analytique et spectrale")
plt.legend()
plt.grid(True)
plt.show()

## Partie 4 
H0 = 1.0

# Calcul du multiplicateur de Fourier H
with np.errstate(divide='ignore', invalid='ignore'):
    H_hat = -1j * np.sign(k) * np.tanh(H0 * np.abs(k))
    H_hat[np.isnan(H_hat)] = 0

# Termes à comparer
v = ft 
v_hat = np.fft.fft(v)

term1_hat = -(1/H0) * H_hat * v_hat
term1 = np.fft.ifft(term1_hat).real 

dv_spec = np.fft.ifft(1j * k * v_hat).real

## Partie 5
g = 9.81
T_max = 10.0
dt = 0.05
nt = int(T_max / dt)

zeta = np.exp(-0.25 * (x - L/2)**2) 
v = 0 * x                           

# Passage en Fourier
zeta_hat = np.fft.fft(zeta)
v_hat = np.fft.fft(v)

# Pour stocker la solution Shallow Water 
zeta_sw_hat = zeta_hat.copy()
v_sw_hat = v_hat.copy()

def deriv(z_h, v_h, mode='DN'):
    """Calcule les dérivées temporelles dans l'espace de Fourier"""
    if mode == 'DN': 
        dz_dt = H_hat * v_h
    else:            
        dz_dt = -H0 * (1j * k) * v_h
    
    dv_dt = -g * z_h
    return dz_dt, dv_dt

# --- Boucle temporelle (Runge-Kutta 4) ---
for _ in range(nt):
    # On itère sur les deux modèles pour les comparer
    for label in ['DN', 'SW']:
        zh = zeta_hat if label == 'DN' else zeta_sw_hat
        vh = v_hat if label == 'DN' else v_sw_hat
        
        # RK4 step
        k1_z, k1_v = deriv(zh, vh, mode=label)
        k2_z, k2_v = deriv(zh + 0.5*dt*k1_z, vh + 0.5*dt*k1_v, mode=label)
        k3_z, k3_v = deriv(zh + 0.5*dt*k2_z, vh + 0.5*dt*k2_v, mode=label)
        k4_z, k4_v = deriv(zh + dt*k3_z, vh + dt*k3_v, mode=label)
        
        if label == 'DN':
            zeta_hat += (dt/6) * (k1_z + 2*k2_z + 2*k3_z + k4_z)
            v_hat    += (dt/6) * (k1_v + 2*k2_v + 2*k3_v + k4_v)
        else:
            zeta_sw_hat += (dt/6)














