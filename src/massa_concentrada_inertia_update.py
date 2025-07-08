# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:16:28 2025

@author: valeria.luz
"""

import numpy as np
import matplotlib.pyplot as plt

# Função de forma e sua derivada segunda
def phi(x):
    return (x / L)**2 - (x / L)**3

def d2phi_dx2(x):
    return (2 / L**2) - (6 * x / L**3)

# Parâmetros da viga
L = 1.290 # m (comprimento da viga)
E = 2.1e11 # Pa (Módulo de elasticidade do aço)
b = 0.156 # m (largura da viga)
h = 0.0415 # m (altura da viga)


rho = 7800 # kg/m³ (densidade do aço)
m1 = 52186e-3
V1 = 20663970e-9
densidade1 = m1 / V1  # kg/m³
rho = densidade1  # kg/m³ 

A = b * h # m² área da seção transversal
I = (b * h**3) / 12 # m^4 (momento de inércia da seção transversal)
mu = rho * A  # massa por metro

# Massas de cada elo
m = [1.34, 1.31, 1.97, 4.75, 2.35, 4.65]  # kg

# Momentos de inércia de cada elo
h1 = 156e-3  # m (altura do elo 1)
b1 = 128e-3  # m (largura do elo 1)

h2 = 138e-3  # m (altura do elo 2)
b2 = 120e-3  # m (largura do elo 2)

h3 = 120e-3  # m (altura do elo 3)
b3 = 130e-3  # m (largura do elo 3)

h4 = 128e-3  # m (altura do elo 4)
b4 = 128e-3  # m (largura do elo 4)

h5 = 114e-3  # m (altura do elo 5)
b5 = 130e-3  # m (largura do elo 5)

h6 = 89e-3  # m (altura do elo 6)
b6 = 98e-3  # m (largura do elo 6)

I_link1 = np.array([74482496, 18377028, 26697058, 4498725, 4498725, 4498725, 4498725,
           4498725, 8485685, 9062566, 648208, 449134, 303528, 303528,
           303528, 303528])*1e-12  # m^4
x_link1 = np.linspace(0,220, len(I_link1), dtype=int) # m

U1 = 0
for i in range(len(I_link1)-1):
    U1 += 0.5 * E * I_link1[i] * (x_link1[i+1] - x_link1[i]) * (2 / L**2 - 6 * (x_link1[i] / L)**3)**2
print(f"U1 sbagliato: {U1:.2f} J")

for i in range(len(I_link1)-1):
    a = np.array([x_link1[i], x_link1[i+1]])
    U1 += 0.5 * E * I_link1[i] * np.trapezoid(d2phi_dx2(a)**2, a)
print(f"U1 corretto: {U1:.2f} J")


# Massas concentradas
massas = [1.34, 1.31, 1.97, 4.75, 2.35, 4.65]  # kg
massas.reverse()

posicoes = [0.205, 0.487, 0.731, 0.947, 1.168, 1.290]  # m

print(f"Densidade do aço: {rho:.2f} kg/m³")
m1 = 9998e-3
V1 = 3408258e-9
m1 = 52186e-3
V1 = 20663970e-9
densidade1 = m1 / V1  # kg/m³
print(f"Densidade do primeiro elo: {densidade1:.2f} kg/m³")


# Energia potencial (flexão)
x_vals = np.linspace(0, L, 1000)
# U = 0.5 * E * I * np.trapezoid(d2phi_dx2(x_vals)**2, x_vals) #trapz is deprecated, using trapezoid instead

U_og = 0.5 * E * I * np.trapezoid(d2phi_dx2(x_vals)**2, x_vals) #trapz is deprecated, using trapezoid instead
x_rest = np.linspace(250, L, 1000)
U_rest = 0.5 * E * I * np.trapezoid(d2phi_dx2(x_rest)**2, x_rest) #trapz is deprecated, using trapezoid instead
U1_calc = U_og - U_rest

U = U_og

print(f"U original: {U_og:.2f} J")
print(f"U rest: {U_rest:.2f} J")
print(f"U1 aproximado: {U1_calc:.2f} J")
print(f"U1 variável: {U1:.2f} J")

# Energia cinética total
T_cont = 0.5 * mu * np.trapezoid(phi(x_vals)**2, x_vals) #trapz is deprecated, using trapezoid instead
T_massas = 0.5 * sum(m * phi(x)**2 for m, x in zip(massas, posicoes))
T = T_cont + T_massas

# Frequência natural aproximada
f = (1 / (2 * np.pi)) * np.sqrt(U / T)

print(f"Frequência natural aproximada com 6 massas concentradas: {f:.2f} Hz")

plt.figure(figsize=(8, 4))
plt.plot(x_vals, phi(x_vals), label='Função de forma (modo aproximado)')
plt.scatter(posicoes, [phi(x) for x in posicoes], color='red', label='Massas concentradas')
plt.xlabel('x [m]')
plt.ylabel('φ(x)')
plt.title('Modo aproximado de vibração com as massas dos redutores')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ======================================================================================================
# Se baseando no Rayleigh's methods, porém calculando as energias seguindo
# o estudo realizado no artigo "INFLUENCE OF THE MANIPULATOR CONFIGURATION ON VIBRATION EFFECTS"
# PIETRUŚ, P., & GIERLAK, P. (2020). Influence of the manipulator configuration on vibration effects.
# DOI 10.2478/ama-2023-0060 
# ======================================================================================================

# Mass moment of intertia of the links # kg*m^2
# Retirados do SolidWorks criando o centro de massa da peça
# e posteriormente um sistema de coordenadas no centro de massa
# NOTA: Cuidado com a posição da peça em relação à montagem do sistema de coordenadas
I1 = 358572.990 #kg*mm²
Is2 = 52744.884 #kg*mm²
Is3 = 29123.950 #kg*mm²
Is4 = 85265 #kg*mm²
Is5 = 1145 #kg*mm²
Is6 = 5518 #kg*mm²

m1 = 9803 # g
m2 = 4696 # g
m3 = 3414 # g
m4 = 4018 # g
m5 = 584 # g
m6 = 2868 # g

l2 = 338.24 #mm
l3 = 580.33 #mm
l4 = 823.58 #mm
l5 = 981.82 #mm
l6 = 1046.79 #mm

beta1 = 0
alpha1 = 0
beta2 = 0
alpha2 = 0
beta3 = 0
alpha3 = 0
beta4 = 0
alpha4 = 0
beta5 = 0
alpha5 = 0
beta6 = 0
alpha6 = 0

q1 = beta1 + alpha1
q2 = beta2 + alpha2
q3 = beta3 + alpha3
q4 = beta4 + alpha4
q5 = beta5 + alpha5
q6 = beta6 + alpha6

# E = 1/2*I_i_A*q_i^2
E1 = 0.5 * I1 * q1**2

E2 = 0.5 * m2 * vs2**2 + 0.5 * Is2 * (q1 + q2)**2
E3 = 0.5 * m3 * vs3**2 + 0.5 * Is3 * (q1 + q2 + q3)**2
E4 = 0.5 * m4 * vs4**2 + 0.5 * Is4 * (q1 + q2 + q3 + q4)**2
E5 = 0.5 * m5 * vs5**2 + 0.5 * Is5 * (q1 + q2 + q3 + q4 + q5)**2
E6 = 0.5 * m6 * vs6**2 + 0.5 * Is6 * (q1 + q2 + q3 + q4 + q5 + q6)**2

# Energia total
E_total = E1 + E2 + E3 + E4 + E5 + E6