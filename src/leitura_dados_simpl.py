# -*- coding: utf-8 -*-
"""
Created on Thu May 15 13:38:05 2025

@author: valeria.luz
"""
import pandas as pd                 # biblioteca para importar informações
import matplotlib.pyplot as plt     # biblioteca de gráfico
from scipy.signal import find_peaks # encontrar picos
from statistics import mean, mode   # biblioteca estatística

#Leitura das informações

nome = 'dados/Teste_Vibracao_Livre.lvm'
dado = pd.read_csv(nome, header = 21, sep = '\s+', decimal=",")

dados = dado.drop(columns=['Comment']) #exclusão da coluna de comentários

dados.columns = ['Tempo [s]', 'Aceleracao [m/s2]'] #nomeando as colunas

# Cortando os dados de forma arbitrária
dados_certos = dados.iloc[3000:,:]
dados_certos.reset_index(drop=True, inplace=True) #reordenar os index

#Encontrar os picos
peaks,_ = find_peaks(dados_certos['Aceleracao [m/s2]'], height=0.1)
picos= dados_certos.iloc[peaks]

# Calculo da frequência
freq = []

for i in range(len(picos)-1):
    
    ti = picos.iloc[i, 0]
    tf = picos.iloc[i+1, 0]
    
    print(ti)
    
    freqi = int(1/(tf-ti))
    
    freq.append(freqi)
    
media = mean(freq)
moda = mode(freq)

info = dados_certos['Aceleracao [m/s2]']
    
#plot dos picos
plt.plot(info, label = 'Aceleração')
plt.plot(peaks, info[peaks], "x", label = 'Picos')
plt.xlabel('Tempo [s]')
plt.ylabel('Aceleração [m/s²]')
plt.legend()
plt.show()
    
    
    
    





