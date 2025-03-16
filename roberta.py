import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
class RoBERTa:
    def __init__(self, cd1: str, cd2: str):
        self.cd1 = cd1
        self.cd2 = cd2
        self.model = SentenceTransformer('stsb-roberta-large')

    def similarity(self):
        sentences = [self.cd1, self.cd2]
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return similarity_score
