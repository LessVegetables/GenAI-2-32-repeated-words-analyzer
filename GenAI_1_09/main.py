import os
import sys

import nltk
from nltk import word_tokenize, ngrams
from nltk.probability import FreqDist


def find_ngrams(input_file: str) -> None:
    """
    Находит и выводит топ-5 биграмм и триграмм из текстового файла.

    Args:
        input_file (str): Путь к входному файлу с текстом

    Returns:
        None
    """
    if not os.path.exists(input_file):
        print(f"Ошибка: Файл {input_file} не найден")
        return

    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return

    tokens = word_tokenize(text.lower())

    bigrams = list(ngrams(tokens, 2))
    trigrams = list(ngrams(tokens, 3))

    freq_bigrams = FreqDist(bigrams)
    freq_trigrams = FreqDist(trigrams)

    print("Топ-5 биграмм:")
    for bigram, count in freq_bigrams.most_common(5):
        print(f"{' '.join(bigram)}: {count}")

    print("\nТоп-5 триграмм:")
    for trigram, count in freq_trigrams.most_common(5):
        print(f"{' '.join(trigram)}: {count}")


if __name__ == "__main__":
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    
    if len(sys.argv) > 1:
        input_path = sys.argv[1]
    else:
        print("Ошибка: укажите входной файл")
        exit()
    
    find_ngrams(input_path)