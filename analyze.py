import os
from glob import glob
from typing import List, Dict

import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams, FreqDist
import re

# Garante download dos recursos do NLTK
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Listas de palavras-chave mais completas (artigos em inglês)
KW_OBJETIVO = [
    'objective', 'objectives',
    'objective of this article',
    'aim', 'aims', 'aiming'
    'goal', 'goals',
    'purpose', 'purposes',
    'this paper aims', 'this paper proposes',
    'in this work we', 'in this paper we',
    'the aim of this'
]

KW_PROBLEMA = [
    'problem', 'problems',
    'challenge', 'challenges', 'important',
    'issue', 'issues',
    'difficulty', 'difficulties',
    'limitation', 'limitations',
    'drawback', 'drawbacks',
    'barrier', 'barriers',
    'gap', 'gaps',
    'we address', 'we tackle', 'we face'
]

KW_METODO = [
    'method', 'methods',
    'methodology', 'methodologies',
    'approach', 'approaches',
    'technique', 'techniques',
    'framework', 'frameworks',
    'algorithm', 'algorithms',
    'we propose', 'we present',
    'we develop', 'we use',
    'we implement', 'implementation',
    'based on', 'built upon'
]

KW_CONTRIB = [
    'contribute', 'contributes', 'contributing',
    'this study contributes', 'our contribution',
    'novel', 'novelty',
    'we demonstrate', 'we show',
    'we illustrate', 'we highlight',
    'for the first time',
    'we provide', 'we offer',
    'our findings', 'our results',
    'bring insight', 'shed light'
]
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


# 1. Leitura de PDFs

def carregar_pdfs_diretorio(diretorio: str) -> List[str]:
    """
    Lê todos os arquivos PDF em um diretório e retorna a lista de caminhos.
    """
    return glob(os.path.join(diretorio, '*.pdf'))


def extrair_texto_pdf(caminho: str) -> str:
    texto = []
    with open(caminho, 'rb') as f:
        leitor = PyPDF2.PdfReader(f)
        for pagina in leitor.pages:
            texto.append(pagina.extract_text() or "")
    raw = '\n'.join(texto)
    # desconsidera tudo antes de 'Abstract'
    m = re.search(r'abstract', raw, flags=re.IGNORECASE)
    return raw[m.start():] if m else raw


# 2. Pré-processamento

def preprocessar_texto(texto: str) -> List[str]:
    """
    Tokeniza, converte para minúsculas, remove stopwords e lematiza.
    """
    tokens = nltk.word_tokenize(texto)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lematizador = WordNetLemmatizer()
    return [lematizador.lemmatize(t) for t in tokens]

# 3. Frequência de termos e n-gramas

def obter_termos_frequentes(listas_tokens: List[List[str]], n: int = 10) -> List[tuple]:
    """
    Achata todas as listas de tokens e retorna os n termos mais frequentes.
    """
    plano = [tok for doc in listas_tokens for tok in doc]
    freq = FreqDist(plano)
    return freq.most_common(n)


def obter_n_gramas(listas_tokens: List[List[str]], tamanho: int = 2, n: int = 10) -> List[tuple]:
    """
    Retorna os n-gramas mais frequentes de um dado tamanho.
    """
    todos_ngrams = []
    for tokens in listas_tokens:
        todos_ngrams.extend(list(ngrams(tokens, tamanho)))
    freq = FreqDist(todos_ngrams)
    return freq.most_common(n)

# 4. Extração de referências

def extrair_referencias(texto: str) -> List[str]:
    """
    Extrai as referências bibliográficas dividindo no cabeçalho 'References' ou 'Bibliography'.
    """
    partes = re.split(r'References?|Bibliography', texto, flags=re.IGNORECASE)
    if len(partes) > 1:
        bloco_refs = partes[-1]
        return [linha.strip() for linha in bloco_refs.split('\n') if linha.strip()]
    return []

# 5. Extração de sentenças por seção

def extrair_sentencas_secao(texto: str, palavras_chave: List[str]) -> str:
    """
    Retorna a primeira sentença que contenha qualquer palavra-chave.
    """
    frases = nltk.sent_tokenize(texto)
    for frase in frases:
        for chave in palavras_chave:
            if chave.lower() in frase.lower():
                return frase.strip()
    return ''

# Função principal: pipeline completo

def analisar_corpus(diretorio: str) -> Dict[str, any]:
    """
    Executa pipeline de análise:
    - carrega PDFs
    - extrai texto
    - pré-processa
    - calcula frequências
    - extrai referências e metadados
    """
    caminhos = carregar_pdfs_diretorio(diretorio)
    textos = [extrair_texto_pdf(c) for c in caminhos]
    listas_tokens = [preprocessar_texto(t) for t in textos]

    termos = obter_termos_frequentes(listas_tokens, n=10)
    ngramas = obter_n_gramas(listas_tokens, tamanho=2, n=10)

    metadados = []
    for texto in textos:
        metadados.append({
            'objetivo':        extrair_sentencas_secao(texto, KW_OBJETIVO),
            'problema':        extrair_sentencas_secao(texto, KW_PROBLEMA),
            'metodo':          extrair_sentencas_secao(texto, KW_METODO),
            'contribuicao':    extrair_sentencas_secao(texto, KW_CONTRIB)
        })


    referencias = [extrair_referencias(t) for t in textos]

    return {
        'arquivos': caminhos,
        'termos_frequentes': termos,
        'ngramas_frequentes': ngramas,
        'metadados': metadados,
        'referencias': referencias
    }

# Execução via linha de comando
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise de artigos científicos via NLP.')
    parser.add_argument('diretorio_pdf', help='Diretório contendo arquivos PDF')
    parser.add_argument('--saida', help='Caminho para salvar resultados', default='resultados.txt')
    args = parser.parse_args()

    resultado = analisar_corpus(args.diretorio_pdf)
    with open(args.saida, 'w', encoding='utf-8') as f:
        for idx, meta in enumerate(resultado['metadados']):
            linha = ';;'.join([
                os.path.basename(resultado['arquivos'][idx]),
                meta['objetivo'],
                meta['problema'],
                meta['metodo'],
                meta['contribuicao']
            ])
            f.write(linha + '\n')

    print(f'Resultados salvos em {args.saida}')