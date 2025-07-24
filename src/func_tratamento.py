'''
Tratamento FFT
=====

Conjunto de funções utilizadas para o tratamento
dos dados vibracionais obtidos em laboratório, com
a criação de FFT, analisada inteira ou por corte,
e cálculo das frequências dominantes

Funções
--
#### escolha_arquivo :
    Realiza a leitura de todos os arquivos presentes
    no directory fornecido e permite a escolha de leitura
    de um desses
#### tratamento_FFT:
    Realiza o tratamento da FFT de um arquivo de dados especificado.

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, periodogram
from scipy.ndimage import gaussian_filter1d
import os

def escolha_arquivo(directory:str):
    '''
    Realiza a leitura de todos os arquivos presentes
    no directory fornecido e permite a escolha de leitura
    de um desses
    
    Parameters
    ----------
    directory : str
        Caminho da pasta com todos os arquivos de interesse
    '''
    # Get all files (ignoring folders)
    file_paths = []
    for root, dirs, files in os.walk(directory):
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
    
    return arquivo


def tratamento_FFT(arquivo:str, corte:bool=False,
                   t_i: float = None, t_f:float = None):
    '''
    Realiza o tratamento da FFT de um arquivo de dados especificado.
    
    Parameters
    ----------
    arquivo : str
        Caminho do arquivo contendo os dados.
    corte: bool, optional
        Indica se o corte será aplicado. Default é falso
    t_i: float, optional
        Tempo inicial do corte. Default é None
    t_f: float, optional
        Tempo final do corte. Default é None
    '''
    
    if corte and (t_i is None or t_f is None):
        raise ValueError("Você deve fornecer t_i e t_f quando 'corte' é True.")
    
    # === LEITURA DO ARQUIVO ===
    dados = pd.read_csv(arquivo, sep='\t', skiprows=22, decimal=',')

    # === AJUSTE DE COLUNAS ===
    dados.columns = ['Index', 'X_Value', 'Time', 'Acceleration', 'Comment']
    tempo = dados['Time'].values
    acel = dados['Acceleration'].values

    if corte:
        mascara = (tempo >= t_i) & (tempo <= t_f)
        tempo_corte = tempo[mascara]
        acel_corte = acel[mascara]
    else:
        tempo_corte = tempo
        acel_corte = acel

    # === PARÂMETROS DE AMOSTRAGEM APÓS CORTE ===
    T = np.mean(np.diff(tempo_corte))  # intervalo médio de amostragem
    fs = 1 / T  # frequência de amostragem
    N = len(acel_corte)

    # === NOME DO ARQUIVO PARA USO EM TÍTULOS ===
    nome_arquivo = os.path.splitext(os.path.basename(arquivo))[0]

    # === ESTIMATIVA DE FREQUÊNCIA NATURAL ===
    f_periodo, Pxx = periodogram(acel_corte, fs)
    freq_estimada = f_periodo[np.argmax(Pxx)]
    samples_por_ciclo = int(fs / freq_estimada) if freq_estimada != 0 else N

    # === DETECÇÃO DE PICOS NO DOMÍNIO DO TEMPO ===
    picos_tempo, props = find_peaks(acel,
                                    distance=int(samples_por_ciclo * 0.8),
                                    prominence=np.max(acel)*0.2)

    # === PLOT DO SINAL ORIGINAL COM PONTOS DE PICO ===
    plt.figure(figsize=(12, 5))
    plt.plot(tempo, acel, label='Aceleração [m/s²]')
    plt.plot(tempo[picos_tempo], acel[picos_tempo], 'rx', label='Picos principais')
    if corte:
        plt.axvspan(t_i, t_f, color='yellow', alpha=0.2, label='Intervalo usado na FFT')
    plt.xlabel('Tempo [s]')
    plt.ylabel('Aceleração [m/s²]')
    plt.title(f'{nome_arquivo} - Sinal de Aceleração com Picos Principais')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === FFT COM SINAL CORTADO ===
    fft_vals = np.fft.fft(acel_corte - np.mean(acel_corte))  # remove média
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
        prominence=0.001,
        distance=5
    )

    frequencias_dominantes = fft_freqs[peaks]
    magnitudes_dominantes = fft_mags_smooth[peaks]

    # === ORDENA E LIMITA TOP N FREQUÊNCIAS ===
    ordenado = np.argsort(magnitudes_dominantes)[::-1]
    frequencias_dominantes = frequencias_dominantes[ordenado]
    magnitudes_dominantes = magnitudes_dominantes[ordenado]

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
    if corte:
        intervalo = f" (corte de {t_i} s a {t_f} s)"
    else:
        intervalo = ""
    plt.title(f'{nome_arquivo} - FFT do Sinal de Aceleração{intervalo}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === IMPRIME FREQUÊNCIAS DETECTADAS ===
    if corte:
        intervalo = f" (intervalo {t_i} s a {t_f} s)"
    else:
        intervalo = ""
    print(f"\nFrequências naturais dominantes detectadas em {nome_arquivo}{intervalo}:")
    for f in frequencias_dominantes:
        print(f" - {f:.2f} Hz")
