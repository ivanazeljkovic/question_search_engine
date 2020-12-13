import numpy as np
from typing import *

from settings import logger
from search_engine.vectorizer.tf_idf_vectorizer import TfIdfVectorizer
from search_engine.similarity_scorer.similarity_metrics import cosine_similarity


class QuestionSearchEngine:
    """Search engine for QnA.

    Attributes:
        _corpus (Sequence[str]): Raw question corpus
        _tf_idf_vectorizer (TfIdfVectorizer): Vectorizer that uses Tf-Idf scores
    """

    def __init__(self, questions: Sequence[str], fit_vectorizer: bool = True) -> None:
        """Initialize search engine by creating Tf-Idf vectorizer and vectorizing question corpus.

        Args:
            questions: sequence of raw question corpus
            fit_vectorizer: flag that indicates if fit of Tf-Idf vectorizer mandatory

        Returns:
            no value
        """
        self._corpus = questions
        self._tf_idf_vectorizer = TfIdfVectorizer(use_cache=True)
        if fit_vectorizer:
            self._tf_idf_vectorizer.fit(questions)

    def most_similar(self, query: str, n: int = 5) -> List[Tuple[float, str]]:
        """Find top n most similar questions from corpus, using cosine similarity as score.

        Args:
            query: raw questions input from the user
            n: number of similar questions that should be found

        Returns:
            The list of top n most similar questions from corpus with similarity scores.
        """
        logger.info(f'Matching top {n} similar questions for question "{query}" started')
        vectorized_query = self._tf_idf_vectorizer.transform([query])

        cosine_similarity_scores = cosine_similarity(vectorized_query, self._tf_idf_vectorizer.questions)
        question_ids = np.asarray(range(len(self._tf_idf_vectorizer.questions)))

        nonzero_indices = np.nonzero(cosine_similarity_scores)[0]
        num_of_nonzeros = nonzero_indices.shape[0]

        if num_of_nonzeros == 0:
            logger.info(f'Matching top {n} similar questions done - 0 results')
            return []
        else:
            # filter nonzero cosine similarity scores
            cosine_similarity_scores = np.take(cosine_similarity_scores, nonzero_indices)
            question_indices = np.take(question_ids, nonzero_indices)
            if num_of_nonzeros > n:
                # take top n cosine similarity scores
                high_scores_indices = np.argsort(cosine_similarity_scores)[-n:]
                cosine_similarity_scores = np.take(cosine_similarity_scores, high_scores_indices)
                question_indices = np.take(question_indices, high_scores_indices)

            cosine_similarity_scores = [np.round(score, 4) for score in cosine_similarity_scores]
            questions = [self._corpus[idx] for idx in question_indices]
            result = list(zip(cosine_similarity_scores, questions))
            logger.info(f'Matching top {n} similar questions done - {len(result)} results')

        return sorted(result, key=lambda pair: pair[0], reverse=True)
