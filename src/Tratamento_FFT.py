import os
import pandas as pd
import func_tratamento as ft

# Specify the directory path
directory_path = "Exp_data"  # e.g., "C:/Users/YourName/Documents"

arquivo = ft.escolha_arquivo(directory_path)

ft.tratamento_FFT(arquivo=arquivo,corte=True,t_i=1,t_f=2)

ft.tratamento_FFT(arquivo=arquivo,corte=False)