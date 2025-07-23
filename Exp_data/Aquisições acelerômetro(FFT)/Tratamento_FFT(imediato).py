# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 08:38:57 2025

@author: eduardo.wisbecki
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, periodogram
from scipy.ndimage import gaussian_filter1d
import os

# === LEITURA DO ARQUIVO ===
arquivo = "Teste_FFT_5(1000).lvm"  # Substitua pelo nome do seu arquivo
dados = pd.read_csv(arquivo, sep='\t', skiprows=22, decimal=',')

# === AJUSTE DE COLUNAS ===
dados.columns = ['Index', 'X_Value', 'Time', 'Acceleration', 'Comment']
tempo = dados['Time'].values
acel = dados['Acceleration'].values

# === PARÂMETROS DE AMOSTRAGEM ===
T = np.mean(np.diff(tempo))  # intervalo médio de amostragem
fs = 1 / T  # frequência de amostragem
N = len(acel)

# === NOME DO ARQUIVO PARA USO EM TÍTULOS ===
nome_arquivo = os.path.splitext(os.path.basename(arquivo))[0]

# === ESTIMATIVA DE FREQUÊNCIA NATURAL ===
f_periodo, Pxx = periodogram(acel, fs)
freq_estimada = f_periodo[np.argmax(Pxx)]
samples_por_ciclo = int(fs / freq_estimada)

# === DETECÇÃO DE PICOS NO DOMÍNIO DO TEMPO ===
picos_tempo, props = find_peaks(acel,
                                distance=int(samples_por_ciclo * 0.8),
                                prominence=np.max(acel)*0.2)

# === PLOT DO SINAL NO TEMPO ===
plt.figure(figsize=(12, 5))
plt.plot(tempo, acel, label='Aceleração [m/s²]')
plt.plot(tempo[picos_tempo], acel[picos_tempo], 'rx', label='Picos principais')
plt.xlabel('Tempo [s]')
plt.ylabel('Aceleração [m/s²]')
plt.title(f'{nome_arquivo} - Sinal de Aceleração com Picos Principais')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === FFT ===
fft_vals = np.fft.fft(acel)
fft_freqs = np.fft.fftfreq(N, T)
pos_mask = fft_freqs > 0
fft_freqs = fft_freqs[pos_mask]
fft_mags = np.abs(fft_vals[pos_mask]) * 2 / N

# === FAIXA DE INTERESSE E SUAVIZAÇÃO ===
faixa_util = fft_freqs < 100
fft_freqs = fft_freqs[faixa_util]
fft_mags = fft_mags[faixa_util]
fft_mags_smooth = gaussian_filter1d(fft_mags, sigma=2)

# === DETECÇÃO DE VÁRIOS PICOS RELEVANTES ===
peaks, propriedades = find_peaks(
    fft_mags_smooth,
    prominence=0.001,  # Ajuste conforme o ruído do seu sinal
    distance=5
)

frequencias_dominantes = fft_freqs[peaks]
magnitudes_dominantes = fft_mags_smooth[peaks]

# === ORDENA DO MAIOR PARA O MENOR (opcional limitar top 5) ===
ordenado = np.argsort(magnitudes_dominantes)[::-1]
frequencias_dominantes = frequencias_dominantes[ordenado]
magnitudes_dominantes = magnitudes_dominantes[ordenado]

# Limitar top N frequências dominantes (opcional)
top_N = 5
frequencias_dominantes = frequencias_dominantes[:top_N]
magnitudes_dominantes = magnitudes_dominantes[:top_N]

# === PLOT DA FFT ===
plt.figure(figsize=(12, 5))
plt.plot(fft_freqs, fft_mags_smooth, label='FFT (suavizada)', color='blue')
plt.plot(frequencias_dominantes, magnitudes_dominantes, 'ro', label='Frequências dominantes')

for f, m in zip(frequencias_dominantes, magnitudes_dominantes):
    plt.annotate(f"{f:.2f} Hz", (f, m), textcoords="offset points", xytext=(0, 5), ha='center')

plt.xlabel('Frequência [Hz]')
plt.ylabel('Magnitude')
plt.title(f'{nome_arquivo} - FFT do Sinal de Aceleração')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === IMPRIME FREQUÊNCIAS DETECTADAS ===
print(f"\nFrequências naturais dominantes detectadas em {nome_arquivo}:")
for f in frequencias_dominantes:
    print(f" - {f:.2f} Hz")



