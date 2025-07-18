import os
import re
import PyPDF2
import nltk
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from analyze import analisar_corpus, extrair_texto_pdf, preprocessar_texto

# --------------- Configurações -----------------
PDF_DIR = 'pdfs'              # diretório dos PDFs de Affective Computing
SAIDA_DIR = 'graficos'        # diretório para salvar figuras
os.makedirs(SAIDA_DIR, exist_ok=True)

# palavras-chave para 'Future Work'
future_keywords = [
    'future work', 'future research',
    'directions for future', 'future directions'
]

# --------------- Carregamento do corpus -----------------
resultado = analisar_corpus(PDF_DIR)
pdf_paths = resultado.get('arquivos', [])
raw_texts = []
all_tokens = []
doc_terms_count = {}
doc_years = {}

for path in pdf_paths:
    # extrai texto a partir do abstract
    raw = extrair_texto_pdf(path)
    # remove rodapé/licença
    lines = raw.splitlines()
    filtered = [l for l in lines if not re.search(r'(licen|universidade)', l, flags=re.IGNORECASE)]
    text = "\n".join(filtered)
    raw_texts.append(text)

    # pré-processa tokens (tokenização, stopwords, lematização)
    tokens = preprocessar_texto(text)
    all_tokens.extend(tokens)
    doc_terms_count[path] = Counter(tokens)

    # extrai ano de criação via metadata
    reader = PyPDF2.PdfReader(path)
    year = 'Unknown'
    if reader.metadata and reader.metadata.get('/CreationDate'):
        m = re.search(r'D:(\d{4})', reader.metadata['/CreationDate'])
        if m:
            year = m.group(1)
    doc_years[path] = year

# --------------- Extração de técnicas via regex -----------------
# padrão: captura até 2 palavras antes do sufixo de método
phrase_pattern = r"\b(?:[A-Za-z]+(?: [A-Za-z]+){0,2}) (?:recognition|analysis|classification|detection|processing|tracking)\b"
phrase_counter = Counter()
for text in raw_texts:
    matches = re.findall(phrase_pattern, text, flags=re.IGNORECASE)
    phrase_counter.update([m.lower() for m in matches])
# seleciona as top 10 técnicas
techniques = [p for p,_ in phrase_counter.most_common(10)]
print('Técnicas extraídas:', techniques)

# --------------- Contagem de presença por artigo -----------------
tech_counter = Counter()
for text in raw_texts:
    low = text.lower()
    for tech in techniques:
        if re.search(r'\b' + re.escape(tech) + r'\b', low):
            tech_counter[tech] += 1

# --------------- Captura de termos em 'Future Work' -----------------
future_tokens = []
for text in raw_texts:
    for sent in nltk.sent_tokenize(text):
        if any(kw in sent.lower() for kw in future_keywords):
            future_tokens.extend(preprocessar_texto(sent))

# --------------- Geração e salvamento de gráficos -----------------

# 1) Técnicas: número de artigos que mencionam cada técnica
items = [(tech, tech_counter.get(tech, 0)) for tech in techniques]
if items:
    labels, values = zip(*items)
    plt.figure(figsize=(10,6))
    plt.bar(labels, values)
    plt.title('Número de artigos que mencionam cada técnica')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA_DIR, 'techniques_frequency.png'))
    plt.close()

# 2) Termos mais frequentes: top 20
term_counter = Counter(all_tokens)
top_terms = term_counter.most_common(20)
if top_terms:
    terms, freqs = zip(*top_terms)
    plt.figure(figsize=(10,6))
    plt.bar(terms, freqs)
    plt.title('Top 20 termos mais frequentes')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA_DIR, 'top_terms.png'))
    plt.close()

# 3) Nuvem de palavras
top_dict = dict(top_terms)
if top_terms:
    wc = WordCloud(width=800, height=400, background_color='white')
    wc.generate_from_frequencies(top_dict)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA_DIR, 'wordcloud_general.png'))
    plt.close()

# 4) Evolução temporal dos 5 termos mais frequentes
years = sorted({y for y in doc_years.values() if y != 'Unknown'})
top5 = [t for t,_ in top_terms[:5]]
evolution = defaultdict(list)
for y in years:
    for term in top5:
        count = sum(cnts[term] for p,cnts in doc_terms_count.items() if doc_years[p] == y)
        evolution[term].append(count)
plt.figure(figsize=(8,6))
for term, counts in evolution.items():
    plt.plot(years, counts, marker='o', label=term)
plt.title('Evolução temporal dos 5 termos mais frequentes')
plt.xlabel('Ano')
plt.ylabel('Frequência')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(SAIDA_DIR, 'temporal_evolution.png'))
plt.close()

# 5) Top termos em 'Future Work'
top_future = Counter(future_tokens).most_common(20)
if top_future:
    f_terms, f_freqs = zip(*top_future)
    plt.figure(figsize=(10,6))
    plt.bar(f_terms, f_freqs)
    plt.title('Top termos em "Future Work"')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(SAIDA_DIR, 'future_work_terms.png'))
    plt.close()