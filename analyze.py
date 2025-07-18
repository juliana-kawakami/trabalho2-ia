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
nltk.download('stopwords')
nltk.download('wordnet')

# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# Padrões com pesos para extração de sentenças
OBJ_WEIGHTS = {
    r"\bobjective of this article\b": 5,
    r"\bthis paper aims\b": 5,
    r"\bthis paper proposes\b": 5,
    r"\bthe aim of this\b": 5,
    r"\bin this work we\b": 4,
    r"\bin this paper we\b": 4,
    r"\bwe (aim|propose|present)\b": 3,
    r"\bobjective(s)?\b": 2,
    r"\bgoal(s)?\b": 2,
    r"\bpurpose(s)?\b": 2,
    r"\baim(s|ing)?\b": 1,
}

PROB_WEIGHTS = {
    r"\bthe (main )?problem\b": 5,
    r"\bwe (address|tackle|face)\b": 4,
    r"\b(challenge|issue|difficulty|limitation|drawback|barrier|gap)(s)?\b": 3,
    r"\b(challenges|issues|difficulties|limitations|drawbacks|barriers|gaps)\b": 2,
    r"\bproblem(s)?\b": 1,
}

MET_WEIGHTS = {
    r"\bmethodology\b": 5,
    r"\bwe (implement|develop|use)\b": 4,
    r"\bapproach(es)?\b": 4,
    r"\btechnique(s)?\b": 3,
    r"\balgorithm(s)?\b": 3,
    r"\bframework(s)?\b": 2,
    r"\bbased on\b": 2,
    r"\bmethod(s)?\b": 1,
}

CONTRIB_WEIGHTS = {
    r"\bthis study contributes\b": 5,
    r"\bour contribution\b": 5,
    r"\bfor the first time\b": 4,
    r"\bnovel(ty)?\b": 4,
    r"\bwe (demonstrate|show|illustrate|highlight)\b": 3,
    r"\bour (findings|results)\b": 2,
    r"\bcontribut(e|ing|ion|ions)\b": 1,
}
# –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––


def carregar_pdfs_diretorio(diretorio: str) -> List[str]:
    """
    Lê todos os arquivos PDF em um diretório.
    """
    return glob(os.path.join(diretorio, '*.pdf'))


def extrair_texto_pdf(caminho: str) -> str:
    """
    Extrai texto do PDF, descarta até 'Abstract' e normaliza quebras.
    """
    paginas = []
    with open(caminho, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            paginas.append(page.extract_text() or '')
    raw = '\n'.join(paginas)
    m = re.search(r'abstract', raw, flags=re.IGNORECASE)
    if m:
        raw = raw[m.start():]
    return re.sub(r'\s*\n+\s*', ' ', raw)


def preprocessar_texto(texto: str) -> List[str]:
    tokens = nltk.word_tokenize(texto)
    tokens = [t.lower() for t in tokens if t.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(t) for t in tokens]


def obter_termos_frequentes(token_lists: List[List[str]], n: int = 10) -> List[tuple]:
    all_tokens = [tok for doc in token_lists for tok in doc]
    freq = FreqDist(all_tokens)
    return freq.most_common(n)


def obter_n_gramas(token_lists: List[List[str]], size: int = 2, n: int = 10) -> List[tuple]:
    all_ngrams = []
    for tokens in token_lists:
        all_ngrams.extend(ngrams(tokens, size))
    freq = FreqDist(all_ngrams)
    return freq.most_common(n)


def extrair_referencias(texto: str) -> List[str]:
    partes = re.split(r'References?|Bibliography', texto, flags=re.IGNORECASE)
    if len(partes) > 1:
        bloco = partes[-1]
        return [linha.strip() for linha in bloco.splitlines() if linha.strip()]
    return []


def extract_by_weight(text: str, weights: Dict[str, int], threshold: int = 1) -> str:
    """
    Varre sentenças, soma pesos de padrões casados e retorna a melhor.
    """
    best_score = 0
    best_sent = ''
    for sent in nltk.sent_tokenize(text):
        score = sum(w for pat, w in weights.items() if re.search(pat, sent, flags=re.IGNORECASE))
        if score > best_score:
            best_score, best_sent = score, sent.strip()
    return best_sent if best_score >= threshold else ''


def extract_with_fallback(text: str,
                          weights: Dict[str,int],
                          thresholds: List[int] = [5,4,3,2,1]) -> str:
    """
    Tenta extração em thresholds 5,4,3,2,1 até encontrar.
    """
    for thr in thresholds:
        sent = extract_by_weight(text, weights, threshold=thr)
        if sent:
            return sent
    return ''


def analisar_corpus(diretorio: str) -> Dict[str, any]:
    arquivos = carregar_pdfs_diretorio(diretorio)
    textos = [extrair_texto_pdf(c) for c in arquivos]

    token_lists = [preprocessar_texto(t) for t in textos]
    termos = obter_termos_frequentes(token_lists)
    ngramas = obter_n_gramas(token_lists)

    metadados = []
    for t in textos:
        obj = extract_with_fallback(t, OBJ_WEIGHTS)
        prob = extract_with_fallback(t, PROB_WEIGHTS)
        met = extract_with_fallback(t, MET_WEIGHTS)
        cont = extract_with_fallback(t, CONTRIB_WEIGHTS)
        metadados.append({
            'objetivo': obj,
            'problema': prob,
            'metodo': met,
            'contribuicao': cont
        })

    referencias = [extrair_referencias(t) for t in textos]

    return {'arquivos': arquivos,
            'termos_frequentes': termos,
            'ngramas_frequentes': ngramas,
            'metadados': metadados,
            'referencias': referencias}

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Análise de artigos científicos via NLP.')
    parser.add_argument('diretorio_pdf', help='Diretório contendo arquivos PDF')
    parser.add_argument('--saida', help='Arquivo de saída', default='resultados.txt')
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
