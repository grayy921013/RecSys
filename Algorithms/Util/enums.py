from enum import Enum

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
    MOVIELENS_ID = 18
    
    FILTERED_PLOT = 19
    TAGS = 20
