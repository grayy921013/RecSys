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
from CB import CBRecommender, CBAlgorithmTMDB, CBAlgorithmTFIDF, CBAlgorithmBM25, CBAlgorithmJACCARD, CBAlgorithmTMDB, CBAlgorithmTFIDF2


logger = logging.getLogger('root')


def initializer(field_name, similarity_class, algorithms):
    global globalDict
    globalDict = {}
    globalDict['field_name'] = field_name
    globalDict['SimilarityClass'] = similarity_class
    globalDict['algorithms'] = algorithms
    
def parse(row):
    global globalDict
    field_name = globalDict['field_name']
    SimilarityClass = globalDict['SimilarityClass']
    algorithms = globalDict['algorithms']

    id1 = str(int(row[0]))
    id2 = str(int(row[1]))

    similarityItem = SimilarityClass(id1_id=id1, id2_id=id2)

    i = 2
    for algorithm in algorithms:
        algo = algorithm.__name__.lower()
        db_fieldname = ("%s%s" % (field_name, algo))
        setattr(similarityItem, db_fieldname, row[i])
        i += 1

    return similarityItem

def main(movies_ids_tagged, fields, algorithms):
    
    # Get the recommender
    rec = CBRecommender()
    dataset = PostgresDataHandler()
    if algorithms is None:
        algorithms = [
            CBAlgorithmJACCARD(), 
            CBAlgorithmTFIDF(), 
            CBAlgorithmBM25(),
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
        for algo in [CBAlgorithmJACCARD(), CBAlgorithmTFIDF(), CBAlgorithmBM25()]:
            algo_idx += 1
            dataset.set_field_algo(field, algo)

            # Train the recommender using the given algorithm and dataset
            result = rec.train(data, algo)
            logger.info('%s was trained', algo.__name__)

            # Extract the data from the sparse matrix
            rows, cols = result.nonzero()
            rows_movielens = algo.ids[rows]
            cols_movielens = algo.ids[cols]
            scores = result.data
            logger.info('%d relations found', len(scores))
            
            p = None
            db_fieldname = field.name+'_'+algo.__name__

            if algo_idx == 1:
                # Only save the ids of the movies for the first
                # algorithm, because next one will have the same id
                # pre_frame = np.stack((rows_movielens, cols_movielens, scores)).T
                
                pre_frame = np.rec.fromarrays((rows_movielens, cols_movielens, scores), \
                    names=('id1_id','id2_id',db_fieldname))

                p = pandas.DataFrame(pre_frame)
                
                # p = pandas.DataFrame(pre_frame,\
                #                      columns=['movieid1','movieid2',db_fieldname])
            else:
                p = pandas.DataFrame(scores,\
                                     columns=[field.name+'_'+algo.__name__])
            
            # Temporarily save to a local file
            p.to_pickle(algo.__name__)
            
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
            
            for i in range(len(algorithms)):
                algo = algorithms[i]
                logger.info('Merging algorithm: %s', algo.__name__)
                p_s[i] = pandas.read_pickle(algo.__name__)[pickle_batch:pickle_batch+batch_size]
                if p_s[i].shape[0] == 0:
                    # Exit when the next batch of data is empty
                    break

            if p_s[0].shape[0] == 0:
                # Exit when the batch of data is empty
                break

            p_prev = pandas.concat(p_s, axis=1)
            logger.info('Duration Merge: %f', time()-t_merge)

            # Parse panda Dataframe to an array of django object
            initializer(field.name.lower() + '_', dataset.SimilarityClass, algorithms)
            # dataset.batch = list(map(parse, p_prev.values))
            
            # Save to Database
            t_db = time()
            dataset.save_similarity_batch2(p_prev, \
               'mainsite_similarity%s'% field.name, \
               1000)
            # dataset.save_similarity_batch(1000)
            logger.info('Duration DB: %f', time()-t_db)
            logger.info('Duration: %f', time()-t)

            pickle_batch += batch_size
    
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
