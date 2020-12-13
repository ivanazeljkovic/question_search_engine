import numpy as np


def cosine_similarity(query_vectors: np.ndarray, corpus_vectors: np.ndarray) -> np.ndarray:
    """Calculate cosine similarity scores between two numpy arrays.

    Args:
        query_vectors: numpy array of (N, D) shape with Tf-Idf vector representation for one or multiple sequences,
                       where N is number of Tf-Idf vectors and D is dimension of those vectors
        corpus_vectors: numpy array of (M, D) shape with Tf-Idf vector representation for multiple sequences,
                        where M is number of Tf-Idf vectors and D is dimension of those vectors

    Returns:
        Cosine similarity score for each pair of vectors, in form of numpy array of (N, M) shape
    """
    return query_vectors.dot(corpus_vectors.transpose()).flatten()
