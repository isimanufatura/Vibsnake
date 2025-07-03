# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 10:16:28 2025

@author: valeria.luz
"""

import numpy as np
import matplotlib.pyplot as plt

# Parâmetros da viga
L = 1.290 # m (comprimento da viga)
E = 2.1e11 # Pa (Módulo de elasticidade do aço)
b = 0.156 # m (largura da viga)
h = 0.0415 # m (altura da viga)
rho = 7800 # kg/m³ (densidade do aço)
A = b * h # m² área da seção transversal
I = (b * h**3) / 12 # m^4 (momento de inércia da seção transversal)
mu = rho * A  # massa por metro

# Função de forma e sua derivada segunda
def phi(x):
    return (x / L)**2 - (x / L)**3

def d2phi_dx2(x):
    return (2 / L**2) - (6 * x / L**3)

# Massas concentradas
massas = [1.34, 1.31, 1.97, 4.75, 2.35, 4.65]  # kg
massas.reverse()

posicoes = [0.205, 0.487, 0.731, 0.947, 1.168, 1.290]  # m

# Energia potencial (flexão)
x_vals = np.linspace(0, L, 1000)
U = 0.5 * E * I * np.trapezoid(d2phi_dx2(x_vals)**2, x_vals) #trapz is deprecated, using trapezoid instead

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
