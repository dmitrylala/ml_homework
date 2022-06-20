import codecs
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

# import nltk
import pandas as pd
import pymorphy2
import requests
from bs4 import BeautifulSoup
from nltk import sent_tokenize, regexp_tokenize
from tqdm import tqdm

# _ = nltk.download('punkt')


PROCESSES = cpu_count() - 1
train_path, test_path = Path("data/train_groups.csv/"), Path("data/test_groups.csv/")
parsed_dir, content_dir = Path("parsed/"), Path("content/")
url_stopwords_ru = "https://raw.githubusercontent.com/stopwords-iso/stopwords-ru/master/stopwords-ru.txt"


def get_text(url: str, encoding='utf-8', to_lower=True):
    if url.startswith('http'):
        r = requests.get(url)
        if not r.ok:
            r.raise_for_status()
        return r.text.lower() if to_lower else r.text
    elif os.path.exists(url):
        with open(url, encoding=encoding) as f:
            return f.read().lower() if to_lower else f.read()
    else:
        raise Exception('parameter [url] can be either URL or a filename')


stopwords_ru = get_text(url_stopwords_ru).splitlines()


def normalize_tokens(tokens):
    morph = pymorphy2.MorphAnalyzer()
    return [morph.parse(tok)[0].normal_form for tok in tokens]


def remove_stopwords(tokens, stopwords=None, min_length=4):
    if not stopwords:
        return tokens

    stopwords = set(stopwords)
    tokens = [
        tok for tok in tokens
        if tok not in stopwords and len(tok) >= min_length
    ]
    return tokens


def tokenize_n_lemmatize(text: str, stopwords=None, normalize=True, regex=r'(?u)\b\w{4,}\b'):
    words = [
        w for sent in sent_tokenize(text)
        for w in regexp_tokenize(sent, regex)
    ]

    if normalize:
        words = normalize_tokens(words)
    if stopwords:
        words = remove_stopwords(words, stopwords)

    return words


def process_html(doc_id: int) -> None:
    with codecs.open(content_dir / f"{doc_id}.dat", 'r', 'utf-8', errors='replace') as f:
        soup = BeautifulSoup(f, 'lxml')
        words = soup.get_text(separator='\n')

    text = ' '.join(tokenize_n_lemmatize(words, stopwords=stopwords_ru))
    with open(parsed_dir / f"{doc_id}.txt", 'w') as f:
        f.write(text)


def main():
    print(f"Running with {PROCESSES} processes")

    train_groups = pd.read_csv(train_path)
    test_groups = pd.read_csv(test_path)

    docs_ids = pd.concat([train_groups["doc_id"], test_groups["doc_id"]], axis=0)
    pool = Pool(PROCESSES)

    start = time.time()
    for _ in tqdm(pool.imap_unordered(process_html, docs_ids.values), total=len(docs_ids.values)):
        pass
    print(f"Time taken = {time.time() - start:.10f}")


if __name__ == "__main__":
    main()
