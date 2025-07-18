# Trabalho 2 - Introdução à Inteligência Artificial: Análise de Artigos de Computação Afetiva

# Alunos: Juliana Naomi Kawakami (RA130092) e Murilo Boccardo (RA124160)

Este projeto realiza a análise de artigos científicos na área de Computação Afetiva, extraindo informações chave e gerando gráficos de visualização. O primeiro script, `analyze.py`, realiza a análise de textos dos artigos e cria um arquivo de saída. O segundo script, `vizualize.py`, gera gráficos a partir desses dados.

## Pré-requisitos

1. Python 3.x instalado na sua máquina.
2. Bibliotecas necessárias que serão instaladas na etapa de configuração.

## Passos para Execução

### 1. Criação do Ambiente Virtual (venv)

Antes de rodar o projeto, é recomendado criar um ambiente virtual para garantir que as dependências sejam isoladas.

1. **Crie o ambiente virtual**:

   ```bash
   python3 -m venv venv
   ```

2. **Ative o ambiente virtual**:

   - No **Windows**:

     ```bash
     venv\Scripts\activate
     ```

   - No **MacOS/Linux**:

     ```bash
     source venv/bin/activate
     ```

   Após a ativação, você verá `(venv)` no terminal, indicando que o ambiente virtual está ativo.

### 2. Instalação das Dependências

Dentro do ambiente virtual, instale as bibliotecas necessárias utilizando o `pip`:

```bash
pip install PyPDF2 nltk scikit-learn matplotlib wordcloud
```

### 3. Execução do Script de Análise

O primeiro script, **analyze.py**, realiza a análise dos artigos, extraindo informações e gerando um arquivo de saída chamado `resultados.txt`.

Para rodar o script de análise, execute o seguinte comando:

```bash
python analyze.py pdfs --saida resultados.txt
```

Onde:

- `pdfs` é o diretório onde os arquivos PDF dos artigos estão localizados.
- `--saida resultados.txt` define o arquivo de saída onde os resultados da análise serão salvos.

**Importante:** O arquivo `resultados.txt` conterá informações extraídas de cada artigo, sendo essas informações o **nome do artigo**, **objetivo**, **problema**, **método**, e **contribuição** de cada um.

### 4. Execução do Script de Visualização

Após a execução do primeiro script, o próximo passo é gerar os gráficos a partir dos dados extraídos.

Para isso, execute o script **vizualize.py**, que irá criar gráficos a partir do arquivo de saída gerado:

```bash
python vizualize.py
```

Este script irá gerar gráficos, como:

- O número de artigos que mencionam cada técnica.
- Os 20 termos mais frequentes.
- A evolução temporal dos termos ao longo dos anos.
- Gráficos sobre as palavras mais usadas nos termos de "Future Work".

Os gráficos serão salvos no diretório `graficos`.

### Estrutura de Arquivos

A estrutura do seu projeto deve ser semelhante a esta:

```
.
├── analyze.py        # Script que faz a análise dos artigos e gera o resultados.txt
├── vizualize.py      # Script que gera os gráficos a partir do resultados.txt
├── pdfs              # Diretório onde os PDFs dos artigos estão localizados
├── graficos          # Diretório onde os gráficos serão salvos
├── requirements.txt  # Arquivo com as dependências do projeto
└── README.md         # Este arquivo
```

### Observações

- O primeiro arquivo, **analyze.py**, é onde a análise real dos artigos ocorre. Ele extrai as informações dos artigos e cria o arquivo de saída `resultados.txt`.
- O segundo arquivo, **vizualize.py**, é responsável por criar gráficos a partir dos artigos selecionados.
