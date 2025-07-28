import os
import pandas as pd
import func_tratamento as fft
import numpy as np

'''
Código para o uso das funções de tratamento das FFTs
'''

# Especificar o caminho da pasta de leitura
directory_path = "Exp_data"  # ex, "C:/Users/YourName/Documents"

# Chamar a função de leitura e escolha
arquivo = fft.escolha_arquivo(directory_path)

# # Exemplo de função tratada com corte
# fft.tratamento_FFT(arquivo=arquivo,corte=True,t_i=1,t_f=2)

# Exemplo de função tratada sem corte
fft.tratamento_FFT(arquivo=arquivo,corte=False)