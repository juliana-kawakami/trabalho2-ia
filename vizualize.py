#!/usr/bin/env python3
import os
import re
import argparse
import nltk
import PyPDF2
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from analyze import analisar_corpus, extrair_texto_pdf, preprocessar_texto

def main():
    parser = argparse.ArgumentParser(
        description="Visualização dos resultados da análise de artigos (Affective Computing)"
    )
    parser.add_argument(
        '--pdf_dir',
        default='pdfs',
        help='Diretório contendo os arquivos PDF'
    )
    parser.add_argument(
        '--saida_dir',
        default='graficos',
        help='Diretório onde os gráficos serão salvos'
    )
    args = parser.parse_args()

    PDF_DIR = args.pdf_dir
    SAIDA_DIR = args.saida_dir
    os.makedirs(SAIDA_DIR, exist_ok=True)
    print(f'✅ Lendo PDFs de "{PDF_DIR}" → salvando gráficos em "{SAIDA_DIR}"\n')

    # --- Carrega e processa o corpus ---
    resultado = analisar_corpus(PDF_DIR)
    pdf_paths = resultado.get('arquivos', [])
    if not pdf_paths:
        print(f'⚠️  Nenhum PDF encontrado em "{PDF_DIR}".')
        return

    raw_texts = []
    all_tokens = []
    doc_terms_count = {}
    doc_years = {}

    for path in pdf_paths:
        raw = extrair_texto_pdf(path)
        # filtra rodapés/licenças
        lines = raw.splitlines()
        filtered = [
            l for l in lines
            if not re.search(r'(licen|universidade)', l, flags=re.IGNORECASE)
        ]
        text = "\n".join(filtered)
        raw_texts.append(text)

        tokens = preprocessar_texto(text)
        all_tokens.extend(tokens)
        doc_terms_count[path] = Counter(tokens)

        # extrai ano da metadata do PDF
        reader = PyPDF2.PdfReader(path)
        year = 'Unknown'
        if reader.metadata and reader.metadata.get('/CreationDate'):
            m = re.search(r'D:(\d{4})', reader.metadata['/CreationDate'])
            if m:
                year = m.group(1)
        doc_years[path] = year

    # --- 1) Extração de técnicas via regex ---
    phrase_pattern = r"\b(?:[A-Za-z]+(?: [A-Za-z]+){0,2}) " \
                     r"(?:recognition|analysis|classification|detection|processing|tracking)\b"
    phrase_counter = Counter()
    for text in raw_texts:
        matches = re.findall(phrase_pattern, text, flags=re.IGNORECASE)
        phrase_counter.update([m.lower() for m in matches])
    techniques = [p for p, _ in phrase_counter.most_common(10)]
    print('🔍 Técnicas extraídas:', techniques, '\n')

    # contagem de presença por artigo
    tech_counter = Counter()
    for text in raw_texts:
        low = text.lower()
        for tech in techniques:
            if re.search(r'\b' + re.escape(tech) + r'\b', low):
                tech_counter[tech] += 1

    # Gráfico 1: Técnicas
    if techniques:
        labels, values = zip(*[(t, tech_counter.get(t, 0)) for t in techniques])
        plt.figure(figsize=(10, 6))
        plt.bar(labels, values)
        plt.title('Número de artigos que mencionam cada técnica')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(SAIDA_DIR, 'techniques_frequency.png')
        plt.savefig(out)
        plt.close()
        print(f'✔️  Gráfico de técnicas salvo em {out}')
    else:
        print('⚠️  Sem técnicas para plotar.')

    # Gráfico 2: Top 20 termos
    term_counter = Counter(all_tokens)
    top_terms = term_counter.most_common(20)
    if top_terms:
        terms, freqs = zip(*top_terms)
        plt.figure(figsize=(10, 6))
        plt.bar(terms, freqs)
        plt.title('Top 20 termos mais frequentes')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(SAIDA_DIR, 'top_terms.png')
        plt.savefig(out)
        plt.close()
        print(f'✔️  Gráfico de termos mais frequentes salvo em {out}')
    else:
        print('⚠️  Sem termos para plotar.')

    # Gráfico 3: Word Cloud
    if top_terms:
        wc = WordCloud(width=800, height=400, background_color='white')
        wc.generate_from_frequencies(dict(top_terms))
        plt.figure(figsize=(10, 5))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.tight_layout()
        out = os.path.join(SAIDA_DIR, 'wordcloud_general.png')
        plt.savefig(out)
        plt.close()
        print(f'✔️  Nuvem de palavras salva em {out}')
    else:
        print('⚠️  Não foi possível gerar nuvem de palavras.')

    # Gráfico 4: Evolução temporal dos top 5 termos
    years = sorted({int(y) for y in doc_years.values() if y.isdigit()})
    years = [str(y) for y in years]
    if years and top_terms:
        top5 = [t for t, _ in top_terms[:5]]
        evolution = {t: [] for t in top5}
        for y in years:
            for t in top5:
                cnt = sum(cnts[t] for p, cnts in doc_terms_count.items() if doc_years[p] == y)
                evolution[t].append(cnt)
        plt.figure(figsize=(8, 6))
        for t, counts in evolution.items():
            plt.plot(years, counts, marker='o', label=t)
        plt.title('Evolução temporal dos 5 termos mais frequentes')
        plt.xlabel('Ano')
        plt.ylabel('Frequência')
        plt.legend()
        plt.tight_layout()
        out = os.path.join(SAIDA_DIR, 'temporal_evolution.png')
        plt.savefig(out)
        plt.close()
        print(f'✔️  Gráfico de evolução temporal salvo em {out}')
    else:
        print('⚠️  Não foi possível gerar evolução temporal.')

    # Gráfico 5: Top termos em "Future Work"
    future_keywords = [
        'future work', 'future research',
        'directions for future', 'future directions'
    ]
    future_tokens = []
    for text in raw_texts:
        for sent in nltk.sent_tokenize(text):
            if any(kw in sent.lower() for kw in future_keywords):
                future_tokens.extend(preprocessar_texto(sent))

    top_future = Counter(future_tokens).most_common(20)
    if top_future:
        f_terms, f_freqs = zip(*top_future)
        plt.figure(figsize=(10, 6))
        plt.bar(f_terms, f_freqs)
        plt.title('Top termos em "Future Work"')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        out = os.path.join(SAIDA_DIR, 'future_work_terms.png')
        plt.savefig(out)
        plt.close()
        print(f'✔️  Gráfico de termos em "Future Work" salvo em {out}')
    else:
        print('⚠️  Não foram encontrados termos para "Future Work".')

if __name__ == '__main__':
    # Garante que os recursos NLTK estejam disponíveis
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    main()
