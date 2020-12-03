from itertools import combinations
import networkx as nx
from sentence_transformers import SentenceTransformer, util
import numpy as np
from scipy.linalg import norm
from scipy.sparse.csgraph import connected_components
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import razdel
from pprint import pprint


class TextRank:
    def __init__(self, model_name:str):
        self.encoder = SentenceTransformer(model_name)

    def model_similarity(self, hash_vec, idx_1, idx_2):
        u = hash_vec[idx_1]
        v = hash_vec[idx_2]

        return self.cosine_sim(u, v)

    def gen_text_rank_summary(self, text, summary_part=0.1, lower=True):
        # Разбиваем текст на предложения
        sentences = [sentence.text for sentence in razdel.sentenize(text)]
        n_sentences = len(sentences)

        # Токенизируем предложения
        sentences_words = [[token.text.lower() if lower else token.text for token in razdel.tokenize(sentence)] for
                           sentence in sentences]

        # хешируем предложения и их вектора
        hash_vec = {idx:np.mean(self.encoder.encode([' '.join(i)]), axis=0) for idx, i in enumerate(sentences_words)}

        # Для каждой пары предложений считаем близость
        pairs = combinations(range(n_sentences), 2)
        scores = [(i, j, self.model_similarity(hash_vec, i, j)) for i, j in pairs]

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
    def calc_scores(references, predictions, text, only_blue=False):
        if only_blue:
            return corpus_bleu([[r] for r in references], predictions)
        else:
            print()
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

    def get_summary(self, records, summary_part=0.1, lower=True, only_blue=False):
        references = []
        predictions = []
        if type(records) == dict:
            summary = records["summary"]
            summary = summary if not lower else summary.lower()
            references.append(summary)

            text = records["text"]
            predicted_summary = self.gen_text_rank_summary(text, summary_part, lower)
            predictions.append(predicted_summary)

            self.calc_scores(references, predictions, text, only_blue)
        elif type(records) == str:
            predicted_summary = self.gen_text_rank_summary(records, summary_part, lower)
            predictions.append(predicted_summary)
            print('Полный текст:')
            pprint(records, width=150)
            print('-' * 150)
            print("TextRank summary:")
            pprint(predictions[-1], width=150)

    @staticmethod
    def cosine_sim(u, v):
        return np.dot(u, v) / (norm(u) * norm(v))

if __name__ == '__main__':
    pass