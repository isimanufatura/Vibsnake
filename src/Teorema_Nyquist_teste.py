# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 06:20:43 2025

@author: valeria.luz
"""

import numpy as np
import matplotlib.pyplot as plt

# Sinal original: seno de 5 Hz
frequencia_sinal = 5  # Hz
tempo_continuo = np.linspace(0, 1, 1000)  # tempo "contínuo" de 0 a 1 segundo
sinal_original = np.sin(2 * np.pi * frequencia_sinal * tempo_continuo)

fase = np.pi / 3 #direfença de fase pra piorar
#fase = 0

# Amostragem com taxa adequada (20 Hz > 2 * 5 Hz)
taxa_adequada = 20  # Hz
t_adequada = np.linspace(0, 1, int(taxa_adequada) + 1)
s_adequada = np.sin(2 * np.pi * frequencia_sinal * t_adequada + fase)

# Amostragem com taxa insuficiente (6 Hz < 2 * 5 Hz)
taxa_insuficiente = 6  # Hz
t_insuficiente = np.linspace(0, 1, int(taxa_insuficiente) + 1)
s_insuficiente = np.sin(2 * np.pi * frequencia_sinal * t_insuficiente + fase)

# Plotagem
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)


# Sem aliasing
axs[0].plot(tempo_continuo, sinal_original, label='Sinal original (5 Hz)')
axs[0].stem(t_adequada, s_adequada, linefmt='C1-', markerfmt='C1o', basefmt=" ", label='Amostras (20 Hz)')
axs[0].set_title('Amostragem adequada (20 Hz)')
axs[0].legend()
axs[0].grid(True)

# Com aliasing
axs[1].plot(tempo_continuo, sinal_original, label='Sinal original (5 Hz)')
axs[1].stem(t_insuficiente, s_insuficiente, linefmt='C2-', markerfmt='C2o', basefmt=" ", label='Amostras (6 Hz)')
axs[1].set_title('Amostragem insuficiente (6 Hz) → Aliasing')
axs[1].legend()
axs[1].grid(True)

plt.xlabel('Tempo (s)')
plt.tight_layout()
plt.show()
