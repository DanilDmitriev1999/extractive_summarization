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


class LexRank:
    def __init__(self, model_name:str):
        self.encoder = SentenceTransformer(model_name)

    def get_lexrank_summary(self, text, n_sentences_summary=3):
        # Разбиваем текст на предложения
        sentences = [sentence.text for sentence in razdel.sentenize(text)]
        n_sentences = len(sentences)

        # Токенизируем предложения
        sentences_words = [[token.text for token in razdel.tokenize(sentence)] for
                           sentence in sentences]

        embeddings = self.encoder.encode(sentences, convert_to_tensor=True)
        cos_scores = util.pytorch_cos_sim(embeddings, embeddings).numpy()

        centrality_scores = self.degree_centrality_scores(cos_scores)

        most_central_sentence_indices = np.argsort(-centrality_scores)
        predicted_summary = ' '.join(
            [sentences[idx].strip() for idx in most_central_sentence_indices[0:n_sentences_summary]])
        return predicted_summary

    def get_summary(self, records, n_sentences_summary=3, show_full_text=True):
        references = []
        predictions = []

        if type(records) == dict:
            summary = records["summary"]
            references.append(summary)

            text = records["text"]
            predicted_summary = self.get_lexrank_summary(text, n_sentences_summary)
            predictions.append(predicted_summary)

            self.calc_scores(references, predictions, text)
        elif type(records) == str:
            predicted_summary = self.get_lexrank_summary(records, n_sentences_summary)
            predictions.append(predicted_summary)
            if show_full_text:
                print('Полный текст:')
                pprint(records, width=150)
                print('-' * 150)
            print("LextRank summary:")
            pprint(predictions[-1], width=150)

    def degree_centrality_scores(self, similar_matrix, increase_power=True):
        markow_matrix = self.create_markow_matrix(similar_matrix)

        scores = self.stationary_dist(markow_matrix, increase_power=increase_power)

        return scores

    def power_method(self, transition_matrix, increase_power=True):
        eigenvectors = np.ones(len(transition_matrix))

        if len(eigenvectors) == 1:
            return eigenvectors

        transition = transition_matrix.transpose()

        while True:
            eigenvectors_next = np.dot(transition, eigenvectors)

            if np.allclose(eigenvectors_next, eigenvectors):
                return eigenvectors_next

            eigenvectors = eigenvectors_next

            if increase_power:
                transition = np.dot(transition, transition)

    def connected_nodes(self, transition_matrix):
        _, labels = connected_components(transition_matrix)

        groups = []

        for tag in np.unique(labels):
            group = np.where(labels == tag)[0]
            groups.append(group)

        return groups

    def stationary_dist(self, transition_matrix, increase_power=True):
        n_1, n_2 = transition_matrix.shape
        if n_1 != n_2:
            raise ValueError('\'transition_matrix\' should be square')

        distribution = np.zeros(n_1)
        group_idx = self.connected_nodes(transition_matrix)

        for group in group_idx:
            transition_matrix = transition_matrix[np.ix_(group, group)]
            eigenvectors = self.power_method(transition_matrix, increase_power=increase_power)
            distribution[group] = eigenvectors

        return distribution

    def create_markow_matrix(self, similar_matrix):
        n_1, n_2 = similar_matrix.shape
        if n_1 != n_2:
            raise ValueError('\'similar_matrix\' should be square')
        row_sum = similar_matrix.sum(axis=1, keepdims=True)

        return similar_matrix / row_sum

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
            print("LexRank summary:")
            pprint(predictions[-1], width=150)
            print('-' * 150)
            print("BLEU: ", corpus_bleu([[r] for r in references], predictions))

if __name__ == '__main__':
    pass