# Vibsnake

Esse é o repositório para os arquivos do projeto Vib-Snake

## O que é o Vib-Snake?

O Vib-Snake é um projeto que tem como objetivo o **desenvolvimento** de um **procedimento** de análise **computacional** e **experimental** para cadeias robóticas de manipulação, com foco em **vibrações**

## Como o Vib-Snake nasceu?

O projeto de cadeia robótica colaborativa Snake demonstrou algumas vibrações indesejadas durante seu funcionamento, as quais acabaram impactando negativamente no controle e na repetibilidade do movimento.

![Demo](https://media2.giphy.com/media/v1.Y2lkPTc5MGI3NjExaTU2MGs2YmF0emp1Ynh0YmtqeTIxbndvYjQyM2JkdHhlcnBhdmpyMCZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9Zw/cV39IH964Rv7sl8dhB/giphy.gif)

A primeiro ver, a vibração indesejada poderia ser causada por folga nas juntas, porém, pela literatura, sabe-se que as frequências naturais das cadeias robóticas são extremamente baixas, portanto a estrutura poderia estar entrando em ressonância durante sua movimentação. O entendimento desse fenômeno não só é importante para o entendimento do possível _erro_ de projeto, como também uma análise extremamente importante para projetos futuro que utilizam uma cadeia robótica, como por exemplo o escalador e cobot!

Assim, o objetivo do Vib-Snake casa perfeitamente com a necessidade de estudo, sendo então focado para o desenvolvimento da metodologia de projeto e de análise de vibrações de uma cadeia robótica.

## Sumário do repositório

O repositório é dividido nas seguintes pastas:
- 📂 [Dados experimentais](./Exp_data/) → Pasta para arquivos obtidos experimentalmente
- 📝 [Arquivos gerados](./Export/) → Arquivos gerados pelos códigos de análise
- 📓 [Notebooks - Google Colab](./Google_colab/) → Arquivos de código para análise analítica e de dados experimentais em formato .ipynb
- 🧠 [Source Code - Python](./src/) → Arquivos de código para análise analítica e de dados experimentais -❓[Como usar](./src/README.md)