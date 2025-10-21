from nltk import FreqDist, word_tokenize, ngrams
from nltk.corpus import stopwords

from GenAI_1_32.word_analyzer import process_text
from GenAI_1_09.main import find_bi_grams

def main():
    text_path = "war_and_peace.ru.txt"
    text = ""
    with open(text_path, "r") as f:
        text = f.read()

    # tokens = [w.lower() for w in word_tokenize(text) if w.isalpha()]
    # filtered = [w for w in tokens if w not in stopwords.words('english')]

    # --- word frequencies ---
    fdist_words = process_text(text_path, "ru")
    top_words = [w for w, _ in fdist_words.most_common(5)]

    # --- bigram frequencies ---
    fdist_bi = find_bi_grams(text_path, "ru")
    top_bigrams = [' '.join(b) for b, _ in fdist_bi.most_common(5)]

    print("Top words:", top_words)
    print("Top bigrams:", top_bigrams)

    # --- auto comparison ---
    matches = [bg for bg in top_bigrams if any(w in bg for w in top_words)]

    print("\nComparison summary:")
    if matches:
        print("Frequent bigrams containing top words:", matches)
        print("These show recurring \"phrases\" built around key terms.")
    else:
        print("No bigrams overlap with top single words — phrases differ in meaning.")


if __name__ == "__main__":
    main()