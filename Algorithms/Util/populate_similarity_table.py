import sys
import os
import logging
import numpy as np
import pickle
from DataHandler import PostgresDataHandler
from enums import Field

from CB import CBRecommender, CBAlgorithmTMDB, CBAlgorithmTFIDF, CBAlgorithmBM25, CBAlgorithmJACCARD, CBAlgorithmTMDB, CBAlgorithmTFIDF2

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)
logger.handlers = []
FORMAT = '%(message)s'
formatter = logging.Formatter(fmt=FORMAT)
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.propagate = False

logger.warn('Start')

# Load Data
import pandas

def main():
    if os.path.isfile(r'similarity_batch.pkl'):
        pkl_file = open(r'similarity_batch.pkl', 'rb')
        dataset = pickle.load(pkl_file)
        pkl_file.close()
        dataset.clear_similarity()
        dataset.save_similarity_batch()
        # return data1

    movies = pandas.read_csv(r'./Data/ml-20m/movies.csv')
    groundtruth = pandas.read_csv(r'./Data/groundtruth.exp1.csv')
    
    # movies = pandas.read_csv(r'Data\ml-20m\movies.csv')
    # groundtruth = pandas.read_csv(r'Data\groundtruth.exp1.csv')

    groundtruth.shape
    groundtruth.dropna()

    m1 = groundtruth['movieid1'].unique()
    m2 = groundtruth['movieid2'].unique()
    movies_ids_tagged = np.union1d(m1,m2)
    print 'This are the tagged movies ids', len(movies_ids_tagged)

    # Get the recommender
    rec = CBRecommender()

    # Get the connection with the 'database'
    fields = [
        Field.TITLE,\
        Field.GENRE,\
        Field.CAST,\
        Field.PLOT,\
        Field.FULL_PLOT,\
        Field.AWARDS,\
        Field.DIRECTOR,\
        Field.WRITER,\
        Field.LANGUAGE,\
        Field.COUNTRY,\
        # Field.YEAR,\
        # Field.RATING,\
        # Field.RUNTIME,\
        # Field.RELEASED,\
        # Field.METACRITIC,\
        # Field.IMDB_RATING,\
        # Field.IMDB_VOTES
    ]

    dataset = PostgresDataHandler()
    dataset.clear_similarity()
    for field in fields:
        logger.warn('\n%s'% str(field))
        data = dataset.get_data(field, movies_ids_tagged.tolist())
        size = len(data)

        # Single Field
        logger.warn("%d records retrieved", len(data))
        cache = {}
        for algo in [CBAlgorithmJACCARD(), CBAlgorithmTFIDF(), CBAlgorithmBM25()]:
            # Train the recommender using the given algorithm and dataset
            result = rec.train(data, algo)

            print 'Trained Data'

            rows, cols = result.nonzero()

            size = len(rows)

            print 'Similarities', size
            for i in range(size):
                if i % 10000 == 0 and i > 0:
                    print 'Percentage ', float(i)/size

                row = rows[i]
                col = cols[i]
                id1 = algo.ids[row]
                id2 = algo.ids[col]

                score = result[row, col]
                if id1 is None or id2 is None:
                    print id1, id2
                    raise Exception()

                dataset.add_to_similarity_batch(id1, id2, field, algo.__name__, score)

            # Release Memory
            algo.destroy()

    output = open(r'similarity_batch', 'wb')
    pickle.dump(dataset, output, -1)
    output.close()
    dataset.save_similarity_batch()
