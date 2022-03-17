# Análise de BigData e Astroinformática

<!---Esses são exemplos. Veja https://shields.io para outras pessoas ou para personalizar este conjunto de escudos. Você pode querer incluir dependências, status do projeto e informações de licença aqui--->

![GitHub repo size](https://img.shields.io/github/repo-size/joaofrancafisica/BigDataProject?style=for-the-badge)
![GitHub language count](https://img.shields.io/github/languages/count/joaofrancafisica/BigDataProject?style=for-the-badge)
![GitHub forks](https://img.shields.io/github/forks/joaofrancafisica/BigDataProject?style=for-the-badge)
![Bitbucket open issues](https://img.shields.io/bitbucket/issues/joaofrancafisica/BigDataProject?style=for-the-badge)
![Bitbucket open pull requests](https://img.shields.io/bitbucket/pr-raw/joaofrancafisica/BigDataProject?style=for-the-badge)

<img src="cbpf_logo.jpg" alt="logo do cbpf">

> Neste projeto, realizamos a análise de um conjunto de dados de simulação feitos no Software Lenspop (https://github.com/tcollett/LensPop). A análise consiste em duas abordagens: Bayesiana e via Deep Learning. Finalmente, comparamos os dois métodos e investigamos as particularidades de cada método.

## 💻 Pré-requisitos

Antes de começar, verifique se você atendeu aos seguintes requisitos:
<!---Estes são apenas requisitos de exemplo. Adicionar, duplicar ou remover conforme necessário--->
* AutoLens (https://pyautolens.readthedocs.io/en/latest/index.html) versão 2021.8.12.1
* Numpy versão 1.20
* jupyter widgets versão mais recente (no momento, 7.6.5).

## 🚀 Instalando

Para instalar, siga estas etapas:

Linux, macOS e Windows:

Para rodar os algóritmos de forma mais rápida, é necessário realizar o donwload da pasta output (que é a pasta que contém os arquivos de output do pyautolens):
https://drive.google.com/drive/folders/1nAbgyPlrOV7Ow98AmAndiepS6TPh45wI?usp=sharing

Além disso, os dados originais de simulação do LensPop podem ser encontrados em:
https://drive.google.com/drive/folders/1tg9uNw2sB6wy0FUnLiVxrlFPKqIy7Qbe?usp=sharing

## ☕ Utilizando os códigos

Para usar, siga estas etapas:

* Para gerar a análise bayesiana via AutoLens rode o notebook BayesianAnalysis.ipynb
* Para fazer uma análise mais detalhada e comparativa com os modelos de deep learning rode o notebook ResultAnalysis.ipynb
* Para a análise final utilize o ResultAnalysis.ipynb