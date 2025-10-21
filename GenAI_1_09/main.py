import os
import sys

import nltk
from nltk import word_tokenize, ngrams
from nltk.corpus import stopwords
from nltk.probability import FreqDist

# from ..GenAI_1_32.word_analyzer import tokenize_and_filter, lemmatize_words
import pymorphy3
RUSSIAN_EXTRA_STOPWORDS = ['это', 'свой', 'свои', 'весь']

LANGUAGE_EN = 'en'
LANGUAGE_RU = 'ru'
SUPPORTED_LANGUAGES = [LANGUAGE_EN, LANGUAGE_RU]

DEFAULT_ENCODING = 'utf-8'
def tokenize_and_filter(text, language):
    """
    Токенизирует текст, приводит к нижнему регистру и удаляет стоп-слова

    Args:
        text (str): исхооный текст
        language (str): язык текста ('en' или 'ru')

    Returns:
        list: отфильтрованные токены
    """
    word_tokens = [word.lower() for word in word_tokenize(text) if word.isalpha()]

    if language == LANGUAGE_RU:
        stop_words = set(stopwords.words('russian'))
        stop_words.update(RUSSIAN_EXTRA_STOPWORDS)
    else:
        stop_words = set(stopwords.words('english'))

    return [word for word in word_tokens if word not in stop_words]

def lemmatize_words(filtered_tokens, language):
    """
    Лемматизирует список токенов в зависимости от языка

    Args:
        filtered_tokens (list): список отфильтрованных токенов
        language (str): язык текста ('en' или 'ru')

    Returns:
        list: лемматизированные слова
    """
    if language == LANGUAGE_RU:
        morph = pymorphy3.MorphAnalyzer()
        lemmatized_words = []
        for word in filtered_tokens:
            try:
                parsed = morph.parse(word)[0]
                lemmatized_words.append(parsed.normal_form)
            except:
                lemmatized_words.append(word)
        return lemmatized_words
    else:
        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(word) for word in filtered_tokens]


def cleanup_text(text, language) -> list[str]:
    filtered_tokens = tokenize_and_filter(text, language)
    lemmatized_words = lemmatize_words(filtered_tokens, language)
    return lemmatized_words

def get_text_from_file(input_file: str) -> str:
    try:
        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except Exception as e:
        print(f"Ошибка при чтении файла: {str(e)}")
        return

def find_bi_grams(input_file: str, language) -> FreqDist:
    text = get_text_from_file(input_file)

    tokens = cleanup_text(text, language)

    bigrams = list(ngrams(tokens, 2))
    freq_bigrams = FreqDist(bigrams)
    return freq_bigrams

def find_tri_grams(input_file: str, language) -> FreqDist:
    text = get_text_from_file(input_file)

    tokens = cleanup_text(text, language)

    trigrams = list(ngrams(tokens, 3))
    freq_trigrams = FreqDist(trigrams)
    return freq_trigrams
    
def find_ngrams(input_file: str, language) -> None:
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

    freq_bigrams = find_bi_grams(input_file, language)
    freq_trigrams = find_tri_grams(input_file, language)

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
        language = sys.argv[2]
    else:
        print("Ошибка: укажите входной файл")
        exit()
    
    find_ngrams(input_path, language)