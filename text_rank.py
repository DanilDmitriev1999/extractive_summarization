from itertools import combinations
import networkx as nx
from sentence_transformers import SentenceTransformer
import numpy as np
from scipy.linalg import norm
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import razdel
from pprint import pprint


class TextRank:
    def __init__(self, model_name:str):
        self.encoder = SentenceTransformer(model_name)

    def model_similarity(self, w1, w2):
        u = np.mean(self.encoder.encode([' '.join(w1)]), axis=0)
        v = np.mean(self.encoder.encode([' '.join(w2)]), axis=0)

        return self.cosine_sim(u, v)

    def gen_text_rank_summary(self, text, summary_part=0.1, lower=True):
        # Разбиваем текст на предложения
        sentences = [sentence.text for sentence in razdel.sentenize(text)]
        n_sentences = len(sentences)

        # Токенизируем предложения
        sentences_words = [[token.text.lower() if lower else token.text for token in razdel.tokenize(sentence)] for
                           sentence in sentences]

        # Для каждой пары предложений считаем близость
        pairs = combinations(range(n_sentences), 2)
        scores = [(i, j, self.model_similarity(sentences_words[i], sentences_words[j])) for i, j in tqdm(pairs)]

        # Строим граф с рёбрами, равными близости между предложениями
        g = nx.Graph()
        g.add_weighted_edges_from(scores)

        # Считаем PageRank
        pr = nx.pagerank(g)
        result = [(i, pr[i], s) for i, s in enumerate(sentences) if i in pr]
        result.sort(key=lambda x: x[1], reverse=True)

        # Выбираем топ предложений
        n_summary_sentences = max(int(n_sentences * summary_part), 1)
        result = result[:n_summary_sentences]

        # Восстанавливаем оригинальный их порядок
        result.sort(key=lambda x: x[0])

        # Восстанавливаем текст выжимки
        predicted_summary = " ".join([sentence for i, proba, sentence in result])
        predicted_summary = predicted_summary.lower() if lower else predicted_summary
        return predicted_summary

    @staticmethod
    def calc_scores(references, predictions, text):
        print("Count:", len(predictions))
        print('Полный текст:')
        pprint(text, width=150)
        print('-' * 150)
        print("Исходное summary:")
        pprint(references[-1], width=150)
        print('-' * 150)
        print("TextRank summary:")
        pprint(predictions[-1], width=150)
        print('-' * 150)
        print("BLEU: ", corpus_bleu([[r] for r in references], predictions))

    def get_summary(self, records, summary_part=0.1, lower=True):
        references = []
        predictions = []
        # for i, record in tqdm(enumerate(records)):
        #     if i >= nrows:
        #         break

        summary = records["summary"]
        summary = summary if not lower else summary.lower()
        references.append(summary)

        text = records["text"]
        predicted_summary = self.gen_text_rank_summary(text, summary_part, lower)
        predictions.append(predicted_summary)

        self.calc_scores(references, predictions, text)

    @staticmethod
    def cosine_sim(u, v):
        return np.dot(u, v) / (norm(u) * norm(v))
