import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
class SBERTquality:
    def __init__(self, cd1: str, cd2: str):
        self.cd1 = cd1
        self.cd2 = cd2
        self.model = SentenceTransformer('all-mpnet-base-v2')

    def similarity(self):
        query_embeddings = self.model.encode([self.cd1])
        passage_embeddings = self.model.encode(self.cd2.tolist())
        similarity_score = self.model.similarity(query_embeddings, passage_embeddings)
        return similarity_score


