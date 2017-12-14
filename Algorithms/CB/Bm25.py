# -*- coding: utf-8 -*-
"""
     BM25 Vectorizer
===================================================

This implementation was made by: BasilBeirouti and at the moment of writing this documentation it was an
non-merged fork of scikit-learn.

I modified the original fork, only to use sklearn as a package instead of assuming being part of sklearn

Original Repository: https://github.com/BasilBeirouti/scikit-learn/tree/mynewfeature
Original Author:  Basil Beirouti
Pull Request: https://github.com/scikit-learn/scikit-learn/pull/6973
"""

# Author: Caleb De La Cruz P. <delacruzp>

import numpy as np
import scipy.sparse as sp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_non_negative, check_array
from sklearn.feature_extraction.text import CountVectorizer

def _document_frequency(X):
    """Count the number of non-zero values for each feature in sparse X."""
    if sp.isspmatrix_csr(X):
        return np.bincount(X.indices, minlength=X.shape[1])
    else:
        return np.diff(sp.csc_matrix(X, copy=False).indptr)


class Bm25Transformer(BaseEstimator, TransformerMixin):
    """Transform a count matrix to a Bm25 weighted representation

    Bm25 is a weighting scheme that differs from tf-idf in two main ways:
        1.) As the frequency of a given term in a document increases, the
            weight of that term approaches the hyperparameter k asymptotically.

        2.) Longer documents are penalized, and the impact of terms contained
            within them is scaled down. The extent to which verbosity is
            penalized
            is controlled by the hyperparameter b {0,1}.  If b=0, verbosity
            is not
            penalized. If b=1, verbosity of a given document is penalized
            exactly
            in proportion to the ratio that document's length and the average
            document length in the corpus.

    The formula implemented here is:

    bm25(t, d) = IDF(t) * f(t, d)*(k+1)/(f(t,d) + k*(1-b+b*|d|/avgdl))

    bm25(t,d): the bm25 weight of term t in document d.

    IDF(t): the inverse document frequency of term t, calculated as:
    log(num_documents/df(t)) where df(t) is the total number of documents in
    which
    term t appears.

    k, b: hyperparmeters

    |d|: the length of document d

    avgdl: the average document length

    Read more in the :ref:`User Guide <text_feature_extraction>`.

    Parameters
    ----------
    smooth_idf : boolean, default=True
        Smooth idf weights by adding one to document frequencies, as if an
        extra document was seen containing every term in the collection
        exactly once. Prevents zero divisions.

    k: float, default=2
        Controls the extent to which increasing the frequency of a given term
        in a given document increases the corresponding bm25 score of that term
        in that document.

    b: float, default=0.75
        Controls the extent to which more verbose documents are penalized.
        0 is no penalization, 1 is full penalization.

    References
    ----------

    .. [Robertson2011] `Stephen Robertson and Hugo Zaragoza (2011). The
    Probabilistic
                        Relevance Framework: BM25 and Beyond`
    """

    def __init__(self, smooth_idf=True, k=2, b=0.75):
        self.smooth_idf = smooth_idf
        self.k = float(k)
        self.b = float(b)

    def fit(self, X, y=None):
        """Learn the idf (n_samples) and beta vectors (n_features)

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts
        """

        check_array(X, accept_sparse=['csr', 'csc'])

        if not sp.issparse(X):
            X = sp.csc_matrix(X)

        n_samples, n_features = X.shape

        # if X is an array of zeros, raise ValueError
        if X.sum() == 0:
            raise ValueError("X is an array of zeros")

        # raise value error if there are negative values in X
        check_non_negative(X, "Bm25Transformer")

        df = _document_frequency(X)

        # perform idf smoothing if required
        df += int(self.smooth_idf)
        n_samples_calc = n_samples + int(self.smooth_idf)

        # log+1 instead of log makes sure terms with zero idf don't get
        # suppressed entirely.
        idf = np.log(float(n_samples_calc) / df) + 1.0
        self._idf_diag = sp.spdiags(idf, diags=0, m=n_features, n=n_features)

        return self

    def transform(self, X, copy=True):
        """Transform a count matrix to a bm25 representation

        Parameters
        ----------
        X : sparse matrix, [n_samples, n_features]
            a matrix of term/token counts

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        vectors : sparse matrix, [n_samples, n_features]
        """

        if hasattr(X, 'dtype') and np.issubdtype(X.dtype, np.float):
            # preserve float family dtype
            X = sp.csr_matrix(X, copy=copy)
        else:
            # convert counts or binary occurrences to floats
            X = sp.csr_matrix(X, dtype=np.float64, copy=copy)

        n_samples, n_features = X.shape

        check_is_fitted(self, '_idf_diag', 'idf vector is not fitted')

        expected_n_features = self._idf_diag.shape[0]

        if n_features != expected_n_features:
            raise ValueError("Input has n_features=%d while the model"
                             " has been trained with n_features=%d" % (
                                 n_features, expected_n_features))

        # sum along rows for document lengths
        lengths = np.array(X.sum(axis=1)).reshape((n_samples,))
        avglen = sum(lengths) / n_samples

        beta = (1 - self.b + self.b * lengths / avglen)
        self._beta_diag = sp.spdiags(beta, diags=0, m=n_samples, n=n_samples,
                                     format='csr')

        weightedtfs = X.copy()
        binary = X.copy()
        binary.data = np.sign(X.data)

        weightedtfs.data = ((self.k + 1) / (self.k * self._beta_diag.dot(
            binary).data / X.data + 1))

        bm25 = weightedtfs.dot(self._idf_diag)
        return bm25

    @property
    def idf_(self):
        if hasattr(self, "_idf_diag"):
            return np.ravel(self._idf_diag.sum(axis=0))
        else:
            return None

    @property
    def beta_(self):
        if hasattr(self, "_beta_diag"):
            return np.ravel(self._beta_diag.sum(axis=0))
        else:
            return None

class Bm25Vectorizer(CountVectorizer):
    """Convert a collection of raw documents to a matrix of BM25 weighted
    features.

     Equivalent to CountVectorizer followed by BM25Transformer.

     Read more in the :ref:`User Guide <text_feature_extraction>`.

     Parameters
     ----------
     input : string {'filename', 'file', 'content'}
         If 'filename', the sequence passed as an argument to fit is
         expected to be a list of filenames that need reading to fetch
         the raw content to analyze.

         If 'file', the sequence items must have a 'read' method (file-like
         object) that is called to fetch the bytes in memory.

         Otherwise the input is expected to be the sequence strings or
         bytes items are expected to be analyzed directly.

     encoding : string, 'utf-8' by default.
         If bytes or files are given to analyze, this encoding is used to
         decode.

     decode_error : {'strict', 'ignore', 'replace'}
         Instruction on what to do if a byte sequence is given to analyze that
         contains characters not of the given `encoding`. By default, it is
         'strict', meaning that a UnicodeDecodeError will be raised. Other
         values are 'ignore' and 'replace'.

     strip_accents : {'ascii', 'unicode', None}
         Remove accents during the preprocessing step.
         'ascii' is a fast method that only works on characters that have
         an direct ASCII mapping.
         'unicode' is a slightly slower method that works on any characters.
         None (default) does nothing.

     analyzer : string, {'word', 'char'} or callable
         Whether the feature should be made of word or character n-grams.

         If a callable is passed it is used to extract the sequence of features
         out of the raw, unprocessed input.

     preprocessor : callable or None (default)
         Override the preprocessing (string transformation) stage while
         preserving the tokenizing and n-grams generation steps.

     tokenizer : callable or None (default)
         Override the string tokenization step while preserving the
         preprocessing and n-grams generation steps.
         Only applies if ``analyzer == 'word'``.

     ngram_range : tuple (min_n, max_n)
         The lower and upper boundary of the range of n-values for different
         n-grams to be extracted. All values of n such that min_n <= n <= max_n
         will be used.

     stop_words : string {'english'}, list, or None (default)
         If a string, it is passed to _check_stop_list and the appropriate stop
         list is returned. 'english' is currently the only supported string
         value.

         If a list, that list is assumed to contain stop words, all of which
         will be removed from the resulting tokens.
         Only applies if ``analyzer == 'word'``.

         If None, no stop words will be used. max_df can be set to a value
         in the range [0.7, 1.0) to automatically detect and filter stop
         words based on intra corpus document frequency of terms.

     lowercase : boolean, default True
         Convert all characters to lowercase before tokenizing.

     token_pattern : string
         Regular expression denoting what constitutes a "token", only used
         if ``analyzer == 'word'``. The default regexp selects tokens of 2
         or more alphanumeric characters (punctuation is completely ignored
         and always treated as a token separator).

     max_df : float in range [0.0, 1.0] or int, default=1.0
         When building the vocabulary ignore terms that have a document
         frequency strictly higher than the given threshold (corpus-specific
         stop words).
         If float, the parameter represents a proportion of documents, integer
         absolute counts.
         This parameter is ignored if vocabulary is not None.

     min_df : float in range [0.0, 1.0] or int, default=1
         When building the vocabulary ignore terms that have a document
         frequency strictly lower than the given threshold. This value is also
         called cut-off in the literature.
         If float, the parameter represents a proportion of documents, integer
         absolute counts.
         This parameter is ignored if vocabulary is not None.

     max_features : int or None, default=None
         If not None, build a vocabulary that only consider the top
         max_features ordered by term frequency across the corpus.

         This parameter is ignored if vocabulary is not None.

     vocabulary : Mapping or iterable, optional
         Either a Mapping (e.g., a dict) where keys are terms and values are
         indices in the feature matrix, or an iterable over terms. If not
         given, a vocabulary is determined from the input documents.

     binary : boolean, default=False
         If True, all non-zero term counts are set to 1. This does not mean
         outputs will have only 0/1 values, only that the tf term in tf-idf
         is binary. (Set idf and normalization to False to get 0/1 outputs.)

     dtype : type, optional
         Type of the matrix returned by fit_transform() or transform().

     smooth_idf : boolean, default=True
         Smooth idf weights by adding one to document frequencies, as if an
         extra document was seen containing every term in the collection
         exactly once. Prevents zero divisions.

    k: float, default=2
        Controls the extent to which increasing the frequency of a given term
        in a given document increases the corresponding bm25 score of that term
        in that document.

    b: float, default=0.75
        Controls the extent to which more verbose documents are penalized.
        0 is no penalization, 1 is full penalization.

     Attributes
     ----------
     vocabulary_ : dict
         A mapping of terms to feature indices.

     idf_ : array, shape = [n_features], or None
         The learned idf vector (global term weights)
         when ``use_idf`` is set to True, None otherwise.

     stop_words_ : set
         Terms that were ignored because they either:

           - occurred in too many documents (`max_df`)
           - occurred in too few documents (`min_df`)
           - were cut off by feature selection (`max_features`).

         This is only available if no vocabulary was given.

     See also
     --------
     Bm25Transformer
        Apply BM25 weighting scheme to a document-term sparse matrix

     CountVectorizer
         Tokenize the documents and count the occurrences of token and return
         them as a sparse matrix

     TfidfTransformer
         Apply Term Frequency Inverse Document Frequency normalization to a
         sparse matrix of occurrence counts.

     Notes
     -----
     The ``stop_words_`` attribute can get large and increase the model size
     when pickling. This attribute is provided only for introspection and can
     be safely removed using delattr or set to None before pickling.
     """

    def __init__(self, input='content', encoding='utf-8',
                 decode_error='strict', strip_accents=None,
                 lowercase=True, preprocessor=None, tokenizer=None,
                 stop_words=None, token_pattern=r"(?u)\b\w\w+\b",
                 ngram_range=(1, 1), analyzer='word',
                 max_df=1.0, min_df=1, max_features=None,
                 vocabulary=None, binary=False, dtype=np.int64,
                 smooth_idf=True, k=2, b=0.75):
        super(Bm25Vectorizer, self).__init__(
            input=input, encoding=encoding, decode_error=decode_error,
            strip_accents=strip_accents, lowercase=lowercase,
            preprocessor=preprocessor, tokenizer=tokenizer, analyzer=analyzer,
            stop_words=stop_words, token_pattern=token_pattern,
            ngram_range=ngram_range, max_df=max_df, min_df=min_df,
            max_features=max_features, vocabulary=vocabulary, binary=binary,
            dtype=dtype)

        self._bm25 = Bm25Transformer(smooth_idf=smooth_idf, k=k, b=b)

    @property
    def smooth_idf(self):
        return self._bm25.smooth_idf

    @smooth_idf.setter
    def smooth_idf(self, value):
        self._bm25.smooth_idf = value

    @property
    def k(self):
        return self._bm25.k

    @k.setter
    def k(self, value):
        self._bm25.k = value

    @property
    def b(self):
        return self._bm25.b

    @b.setter
    def b(self, value):
        self._bm25.b = value

    @property
    def idf_(self):
        return self._bm25.idf_

    @property
    def beta_(self):
        return self._bm25.beta_

    def fit(self, raw_documents, y=None):
        """Learn vocabulary and idf from training set.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        self : Bm25Vectorizer
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._bm25.fit(X)
        return self

    def fit_transform(self, raw_documents, y=None):
        """Learn vocabulary and idf, return term-document matrix.

        This is equivalent to fit followed by transform, but more efficiently
        implemented.

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            bm25-weighted document-term matrix.
        """
        X = super(Bm25Vectorizer, self).fit_transform(raw_documents)
        self._bm25.fit(X)
        # X is already a transformed view of raw_documents so
        # we set copy to False
        return self._bm25.transform(X, copy=False)

    def transform(self, raw_documents, copy=True):
        """Transform documents to document-term matrix.

        Uses the vocabulary and document frequencies (df) learned by fit (or
        fit_transform).

        Parameters
        ----------
        raw_documents : iterable
            an iterable which yields either str, unicode or file objects

        copy : boolean, default True
            Whether to copy X and operate on the copy or perform in-place
            operations.

        Returns
        -------
        X : sparse matrix, [n_samples, n_features]
            bm25-weighted document-term matrix.
        """
        X = super(Bm25Vectorizer, self).transform(raw_documents)
        return self._bm25.transform(X, copy=False)
