import argparse

import numpy as np
import spacy
from spacy.language import Language
from spacy.pipeline import Pipe



### GLOBALS ###
# Choices of combination function: average, sum, first, last, maxpool
COMBINATION_FUNCTION = "average"

class ContextualVectors(Pipe):
    def __init__(self, nlp):
        self._nlp = nlp
        self.combination_function = COMBINATION_FUNCTION ### modify this here for different versions of part 3

    def __call__(self, doc):
        if type(doc) == str:
            doc = self._nlp(doc)
        self.lengths = doc._.trf_data.align.lengths
        self.tensors = doc._.trf_data.tensors
        doc.user_token_hooks["vector"] = self.vector
        return doc

    ### HERE is where vectors are set
    def vector(self, token):
        trf_vector = []
        for len_idx in range(self.lengths[token.i]):
            try:
                trf_vector.append(self.tensors[0][0][token.i+len_idx])
            except IndexError:
                print("Error")
                print(token)
                return None
        trf_vector = np.array(trf_vector)
        return self.combine_vectors(trf_vector)

    def combine_vectors(self, trf_vector):
        if self.combination_function == "first":
            return trf_vector[0]
        if self.combination_function == "last":
            return trf_vector[-1]
        if self.combination_function == "maxpool":
            return np.maximum(trf_vector, axis=0)
        if self.combination_function == "average":
            return np.average(trf_vector, axis=0)
        if self.combination_function == "sum":
            return np.sum(trf_vector, axis=0)


@Language.factory("trf_vector_hook", assigns=["doc.user_token_hooks"])
def create_contextual_hook(nlp, name):
    return ContextualVectors(nlp)

def setup_argparse():
    p = argparse.ArgumentParser()

    return p.parse_args()


if __name__ == "__main__":
    args = setup_argparse()
