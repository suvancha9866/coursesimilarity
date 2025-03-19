import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
class SBERT:
    def __init__(self, cd1: str, cd2: str):
        self.cd1 = cd1
        self.cd2 = cd2
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def similarity(self):
        sentences = [self.cd1, self.cd2]
        embeddings = self.model.encode(sentences)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()#np.dot(embeddings[0], embeddings[1]) / (np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1]))
        return similarity_score


