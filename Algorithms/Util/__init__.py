from enum import Enum
import numpy as np


class Field(Enum):
    TITLE = 1
    GENRE = 2
    CAST = 3
    PLOT = 4
    FULL_PLOT = 5
    AWARDS = 6

    DIRECTOR = 7
    WRITER = 8
    LANGUAGE = 9
    COUNTRY = 10

    YEAR = 11
    RATING = 12
    RUNTIME = 13
    RELEASED = 14
    METACRITIC = 15
    IMDB_RATING = 16
    IMDB_VOTES = 17

def chunking_dot(big_matrix, small_matrix, chunk_size=100):
    # Make a copy if the array is not already contiguous
    small_matrix = np.ascontiguousarray(small_matrix)
    R = np.empty((big_matrix.shape[0], small_matrix.shape[1]))
    for i in range(0, R.shape[0], chunk_size):
        end = i + chunk_size
        R[i:end] = np.dot(big_matrix[i:end], small_matrix)
    return R
