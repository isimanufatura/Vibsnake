import os
import pandas as pd

# Specify the directory path
directory_path = "Exp_data"  # e.g., "C:/Users/YourName/Documents"

# Get all files (ignoring folders)
file_paths = []
for root, dirs, files in os.walk(directory_path):
    for i, file in enumerate(files):
        full_path = os.path.join(root, file)
        file_paths.append([i + 1, full_path])

df = pd.DataFrame(file_paths, columns=["Número", "Arquivo"])
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
    a = df.drop(columns=["Número"])
    print(f"{a} \n")

x = int(input("Forneça o índice do arquivo de leitura: \n"))

arquivo = str(df.loc[x, "Arquivo"])

with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.max_colwidth', None):
    print(f"\nO arquivo escolhido é:\n{arquivo}\n")

import func_tratamento as ft

ft.tratamento_FFT(arquivo=arquivo,corte=True,t_i=1,t_f=2)

ft.tratamento_FFT(arquivo=arquivo,corte=False)