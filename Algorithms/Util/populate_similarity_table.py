import sys
import os
import logging
import numpy as np
import pandas
import pickle
from time import time
from DataHandler import PostgresDataHandler
from .enums import Field
from mainsite.models import SimilarityGenre
from CB import CBRecommender, CBAlgorithmTMDB, CBAlgorithmTFIDF, CBAlgorithmBM25, CBAlgorithmJACCARD, CBAlgorithmTFIDF2


logger = logging.getLogger('root')


def main(movies_ids_tagged = [], fields=None, algorithms=None, k=None):
    if k is None:
        TOP_K = 100
    else:
        TOP_K = k

    # Get the recommender
    rec = CBRecommender()
    dataset = PostgresDataHandler()
    if algorithms is None:
        algorithms = [
            CBAlgorithmJACCARD,
            CBAlgorithmTFIDF, 
            CBAlgorithmBM25,
        ]

    # Get the connection with the 'database'
    if fields is None:
        fields = [
            Field.TITLE,
            Field.GENRE,
            Field.CAST,
            Field.PLOT,
            Field.FULL_PLOT,
            Field.DIRECTOR,
            Field.WRITER,
            Field.LANGUAGE,
            Field.COUNTRY,
            Field.AWARDS,
            Field.FILTERED_PLOT
        ]

    # For each field caculate the similarity
    for field in fields:
        logger.info('%s\n\n'% field.name)
        
        # Get an array of (id,string) for the field requested
        # Filtering only for the movies in the groundtruth
        data = dataset.get_data(field, movies_ids_tagged)
        size = len(data)
        logger.info("%d records retrieved", size)

        # Set variables initial values
        t = time()
        algo_idx = 0
        # TODO: Look for a better way to set the default field
        dataset.set_field_algo(field, CBAlgorithmJACCARD())

        # For each algorithm
        algos = list(map(lambda x: x(), algorithms))
        for algo in algos:
            logger.debug(algo.__name__)
            algo_idx += 1
            dataset.set_field_algo(field, algo)

            # Train the recommender using the given algorithm and dataset
            result = rec.train(data, algo)
            
            # Extract the data from the sparse matrix
            rows, cols = result.nonzero()
            rows_movielens = algo.ids[rows]
            cols_movielens = algo.ids[cols]
            scores = result.data

            # Filter Out 
            mask = rows_movielens != cols_movielens
            rows_movielens = rows_movielens[mask]
            cols_movielens = cols_movielens[mask]
            scores = scores[mask]

            # Only save the ids of the movies for the first
            # algorithm, because next one will have the same id
            db_fieldname = field.name+'_'+algo.__name__
            pre_frame = np.rec.fromarrays((rows_movielens, cols_movielens, scores), \
                names=('id1_id','id2_id',db_fieldname))

            p = pandas.DataFrame(pre_frame)
            
            # Get top K elements for each movieid1 set 1
            p = p \
                .sort_values(by=['id1_id', db_fieldname], ascending=False) \
                .groupby('id1_id') \
                .head(TOP_K) \
                .reset_index(drop=True)
            
            p = p.sort_values(by=['id1_id'], ascending=True)
            
            # Temporarily save to a local file
            logger.info('%s\t %d/%d saved/found', algo.__name__, p.shape[0], len(scores))
            
            # Temporarily save to a local file
            p.to_pickle('Temp/'+algo.__name__)
            
            # Release Memory
            algo.destroy()
        logger.info('Duration IR: %f', time()-t)
            
        pickle_batch = 0
        batch_size = 1000000
        # Delete similarity table for a given field
        dataset.clear_similarity(field)

        while True:
            t_merge = time()
            
            p_prev = None
            p_s = [None]*len(algorithms)
            p_s = None
            end_idx = 0

            # Look for the score for each algorithms
            # the scores were previously saved in a temporary file
            for i in range(len(algorithms)):
                algo = algorithms[i]()
                logger.info('Merge: algorithm %s', algo.__name__)
                
                # If it's the first algorithm we will use him as standard to obtain min and max
                # TO DO: we should get id1, id2 for each file and then use that 
                if i == 0:
                    p_s = pandas.read_pickle('Temp/'+algo.__name__)[pickle_batch:pickle_batch+batch_size]
                    
                    # TODO: Look for a better way to do this. in the case where the first
                    #       algo is something like 1mm*N record and the second is > than that
                    if p_s.shape[0] == 0:
                        # Exit when the batch of data is empty
                        break

                    if p_s.shape[0] == batch_size:
                        # Do not execute on the last batch
                        start = p_s.id1_id.iloc[0]
                        end_idx = batch_size
                        end = p_s.id1_id.iloc[end_idx-1]
                        while p_s.id1_id.iloc[end_idx-1] == end:
                            end_idx -= 1
                        end = p_s.id1_id.iloc[end_idx-1]
                        p_s = p_s[:end_idx]
                    else:
                        end_idx = p_s.shape[0] 
                        start = p_s.id1_id.iloc[0]
                        end = -1

                    pickle_batch = pickle_batch+end_idx
                else:
                    p_s = pandas.read_pickle('Temp/'+algo.__name__)
                    
                    s_idx = p_s.id1_id.searchsorted(start)[0]
                    if end == -1:
                        e_idx = p_s.shape[0]
                    else:
                        e_idx = p_s.id1_id.searchsorted(end, side='right')[0]
                    p_s = p_s[s_idx:e_idx]
                    
                if p_prev is None:
                    p_prev = p_s
                else:
                    p_prev = pandas.merge(p_prev, p_s, how="outer")

            if p_prev is None:
                # Exit when the batch of data is empty
                break

            logger.info('Duration Merge: %f', time()-t_merge)

            # Save to Database
            t_db = time()
            table_name = 'mainsite_similarity%s'% field.name
            dataset.save_similarity_batch2(p_prev, table_name, 1000)
            logger.info('Duration DB: %f', time()-t_db)
            logger.info('Duration: %f', time()-t)
    
    logger.info('DB JOIN: Started')
    t = time()
    # Clear the similarity table
    dataset.clear_similarity()
    # Insert new batch
    dataset.similarity_join();
    logger.info('DB JOIN: Finished\tDuration: %f', time()-t)

def clear_db(fields = None):
    dataset = PostgresDataHandler()
    
    if fields is None:
        fields = [
            Field.TITLE,
            Field.GENRE,
            Field.CAST,
            Field.PLOT,
            Field.FULL_PLOT,
            Field.DIRECTOR,
            Field.WRITER,
            Field.LANGUAGE,
            Field.COUNTRY,
            Field.AWARDS,
        ]

    for field in fields:
        dataset.clear_similarity(field)
