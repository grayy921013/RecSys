import os
import sys
import django
import logging
# from io import BytesIO as StringIO # Python 2.7
from io import StringIO
from progressbar import ProgressBar, Bar, Percentage, Timer
from time import time, sleep

sys.path.append('../')
sys.path.append('../Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
django.setup()


from django.db import connection
from django.db.models import Aggregate, CharField, Value as V
from django.db.models.functions import Concat
from DataHandler.Base import DataHandler
from Util import Field


from django.contrib.postgres.aggregates.general import StringAgg
from mainsite.models import Movie, \
                            SimilarMovie, \
                            Similarity, \
                            SimilarityPair, \
                            SimilarityTitle, \
                            SimilarityGenre, \
                            SimilarityReleased, \
                            SimilarityDirector, \
                            SimilarityWriter, \
                            SimilarityCast, \
                            SimilarityMetacritic, \
                            SimilarityPlot, \
                            SimilarityFull_plot, \
                            SimilarityLanguage, \
                            SimilarityCountry, \
                            SimilarityAwards, \
                            SimilarityLast_updated, \
                            SimilarityFiltered_plot


logger = logging.getLogger(__name__)




class PostgresDataHandler(DataHandler):
    def __init__(self):
        self.batch = []
        self.cache = {}

    def get_class_for_field(self, field=None):
        if field is None:
            SimilarityClass = Similarity
        elif Field.TITLE == field:
            SimilarityClass = SimilarityTitle
        elif Field.GENRE == field:
            SimilarityClass = SimilarityGenre
        elif Field.RELEASED == field:
            SimilarityClass = SimilarityReleased
        elif Field.DIRECTOR == field:
            SimilarityClass = SimilarityDirector
        elif Field.WRITER == field:
            SimilarityClass = SimilarityWriter
        elif Field.CAST == field:
            SimilarityClass = SimilarityCast
        elif Field.METACRITIC == field:
            SimilarityClass = SimilarityMetacritic
        elif Field.PLOT == field:
            SimilarityClass = SimilarityPlot
        elif Field.FULL_PLOT == field:
            SimilarityClass = SimilarityFull_plot
        elif Field.LANGUAGE == field:
            SimilarityClass = SimilarityLanguage
        elif Field.COUNTRY == field:
            SimilarityClass = SimilarityCountry
        elif Field.AWARDS == field:
            SimilarityClass = SimilarityAwards
        elif Field.FILTERED_PLOT == field:
            SimilarityClass = SimilarityFiltered_plot
        return SimilarityClass

    def clear_similarity(self, field=None):
        SimilarityClass = self.get_class_for_field(field)
        similarities = SimilarityClass.objects.all()
        similarities.delete()

    def set_field_algo(self, field, algo):
        SimilarityClass = self.get_class_for_field(field)

        field_name = str(field).split('.')[1].lower()
        db_fieldname = ("%s_%s" % (field_name, algo)).lower()

        self.field = field
        self.SimilarityClass = SimilarityClass
        self.db_fieldname = db_fieldname
        self.i = 0
        
        return SimilarityClass, db_fieldname
    
    def add_to_similarity_batch(self, id1, id2, field, algo, score):
        if (id1, id2) in self.cache:
            s = self.batch[self.cache[(id1, id2)]]
            flag = True
        else:
            s = self.SimilarityClass(
                id1_id=id1, id2_id=id2
            )
            flag = False
            self.cache[(id1, id2)] = len(self.batch)

        setattr(s, self.db_fieldname, score)
        
        if not flag:
            self.batch.append(s)


    def save_similarity_batch2(self, data, table, batch_size=None):
        from django.db import connection
        from contextlib import closing

        stream = StringIO()
        t_db = time()
        data.to_csv(stream,
                    sep='\t',
                    header=False, 
                    index=False,
                    chunksize=batch_size,
                    na_rep='N')
        logger.info('Duration CSV: %f', time()-t_db)
            
        t_db = time()
        stream.seek(0)

        with closing(connection.cursor()) as cursor:
            cursor.copy_from(
                file=stream,
                table=table,
                sep='\t',
                null='N',
                columns=list(data.columns),
                size=8192,
            )
        logger.info('Duration INSERT: %f', time()-t_db)


    def save_similarity_batch(self, batch_size=None):
        if batch_size is not None:
            chunks = [self.batch[x:x+batch_size] for x in range(0, len(self.batch), batch_size)]
        else:
            chunks = [self.batch]

        logger.debug("Size: %d, Chunks %d", len(self.batch), len(chunks))

        bar = ProgressBar(maxval=len(chunks) + 1, \
                          widgets=['DB: Sim_', self.field.name, ' ', Bar('=', '[', ']'), ' ', Percentage(), ' - ', Timer()])
        bar.start()

        counter = 0
        for chunk in chunks:
            self.SimilarityClass.objects.bulk_create(chunk)
            counter += 1
            bar.update(counter)
                
        # SimilarityClass.objects.bulk_create(self.batch, batch_size=None)
        self.batch = []
        self.cache = {}
        bar.finish()

    def save_similarity(self, id1, id2, field, algo, score):
        # TODO: Move this to another function
        field_name = field.name.lower()
        db_fieldname = ("%s_%s" % (field_name, algo)).lower()
        return Similarity.objects.update_or_create(
            id1_id=id1, id2_id=id2, defaults={db_fieldname: score}
        )

    def get_data(self, field, movielens_ids = []):
        # if not isinstance(field, Field):
        #     raise AttributeError("Parameter field should be value of the enum Field")

        if field.name == 'GENRE':
            field_name = 'genres'

            if len(movielens_ids):
                cursor = Movie.objects.filter(movielens_id__in=movielens_ids).values('id').annotate(genre=StringAgg('genres__name', ', '))
            else:
                cursor = Movie.objects.all().values('id').annotate(genre=StringAgg('genres__name', ', '))
            cursor = cursor.values_list('id', 'genre')

        else:
            field_name = field.name.lower()
            if field_name == 'full_plot':
                field_name = 'tags'
            if len(movielens_ids):
                cursor = Movie.objects.filter(movielens_id__in=movielens_ids).values_list('id', field_name)
            else:
                cursor = Movie.objects.all().values_list('id', field_name)
        
        return list(map(lambda x: [x[0], x[1] or ''], cursor))


    def get_features(self, ids=None, features=None, low=None, high=None, flag_db_ids=False):
        logger.debug('Getting features')
        if features is None:
            return Similarity.objects.all()
        elif ids is None:
            #TODO: Analyse if necessary
            t = time()
            solution = []
            fields_str = ','.join(map(lambda x: 'coalesce(%s,0)' % x, features))
            with connection.cursor() as cursor:
                cursor.execute("SELECT m1.movielens_id, \
                                       m2.movielens_id, %s \
                                  FROM mainsite_similarity s \
                                  LEFT JOIN mainsite_movie m1 ON s.id1_id =  m1.id \
                                  LEFT JOIN mainsite_movie m2 ON s.id2_id =  m2.id \
                                 WHERE m1.id >= %d and m1.id < %d \
                                " % (fields_str, low, high))
                solution = cursor.fetchall()

            logger.debug('Time Retrieving Features: %f', time()-t)
            return solution
        else:
            t = time()
            self.save_similarity_pair_batch(ids)
            logger.debug('Time saving in temp table: %f', time()-t)

            t = time()
            solution = []
            fields_str = ','.join(map(lambda x: 'coalesce(s.%s,0)' % x, features))
            if flag_db_ids:
                #TODO: Analyse if this is the best place to do this
                #      or if it should be in a different function
                with connection.cursor() as cursor:
                    cursor.execute("SELECT p.id1_id, p.id2_id, %s \
                                      FROM mainsite_similaritypair p \
                                      LEFT JOIN mainsite_similarity s ON p.id1_id =  s.id1_id AND p.id2_id =  s.id2_id \
                                    " % (fields_str))
                    solution = cursor.fetchall()
            else:
                with connection.cursor() as cursor:
                    cursor.execute("SELECT cast(m1.movielens_id as int) movielens_id, \
                                           cast(m2.movielens_id as int) movielens_id, %s \
                                      FROM mainsite_similaritypair p \
                                      LEFT JOIN mainsite_similarity s ON p.id1_id =  s.id1_id AND p.id2_id =  s.id2_id \
                                      LEFT JOIN mainsite_movie m1 ON p.id1_id =  m1.id \
                                      LEFT JOIN mainsite_movie m2 ON p.id2_id =  m2.id \
                                    " % (fields_str))
                    solution = cursor.fetchall()

            logger.debug('Time Retrieving Features: %f', time()-t)
            return solution

    def get_pairs(self, low=None, high=None):
        t = time()
        solution = []
        with connection.cursor() as cursor:
            cursor.execute("SELECT id1_id, id2_id \
                              FROM mainsite_similarity s \
                             WHERE s.id1_id >= %d and s.id1_id < %d \
                            " % (low, high))
            solution = cursor.fetchall()

        logger.debug('Time Retrieving Features: %f', time()-t)
        return solution
    
    def similarity_join(self):
        '''
        Executes stored procedure in the database 
        to join all the small field by field similarity tables
        into the big joined mainsite_similarity table
        '''
        bar = ProgressBar(widgets=['DB: Similarity Join - ', Timer()])
        bar.start()
        with connection.cursor() as cursor:
           cursor.callproc('SIMILARITY_JOIN', [])
        bar.finish()

    def save_similarity_pair_batch(self, data, batch_size=1000):
        SimilarityPair.objects.all().delete()
        logger.debug('Truncated')
        
        from django.db import connection
        from contextlib import closing
        import pandas as pd

        stream = StringIO()
        t_db = time()
        
        data = pd.DataFrame(data, columns=['id1_id', 'id2_id'], dtype=int)

        data.to_csv(stream,
                    sep='\t',
                    header=False, 
                    index=False,
                    chunksize=batch_size,
                    na_rep='N')
        logger.info('Duration CSV: %f', time()-t_db)
            
        t_db = time()
        stream.seek(0)

        with closing(connection.cursor()) as cursor:
            cursor.copy_from(
                file=stream,
                table='mainsite_SimilarityPair',
                sep='\t',
                null='N',
                columns=list(data.columns),
                size=8192,
            )
        logger.info('Duration INSERT: %f', time()-t_db)
        return None

    def clear_similar_movies(self):
        SimilarMovie.objects.all().delete()

    def save_similar_movies(self, pairs, batch_size=1000):
        batch = list(map(lambda x: SimilarMovie(movie_id = x[0], similar_movie_id=x[1], rank=0, algorithm=0), pairs))
        logger.debug('Parsed')

        if batch_size is not None:
            chunks = [batch[x:x+batch_size] for x in range(0, len(batch), batch_size)]
        else:
            chunks = [batch]
        logger.debug("Size: %d, Chunks %d", len(batch), len(chunks))
        
        bar = ProgressBar(maxval=len(chunks)+1, \
            widgets=['DB: Similar Movie', Bar('=', '[', ']'), ' ', Percentage(), ' - ', Timer()])
        bar.start()
        counter = 0
        for chunk in chunks:
            SimilarMovie.objects.bulk_create(chunk)
            counter += 1
            bar.update(counter)
        bar.finish()


    def save_als(self, data):
        t = time()
        solution = []
        with connection.cursor() as cursor:
            try:
                cursor.execute("DROP TABLE mainsite_similarityals;")
                # solution = cursor.fetchall()
                # print solution
            except Exception as e:
                print(e)
            cursor.execute("""CREATE TABLE public.mainsite_similarityals
                            (
                                id1_id integer,
                                id2_id integer,
                                als_cosine double precision
                            );""")
            
            # solution = cursor.fetchall()
            # print solution

        logger.debug('Time Retrieving Features: %f', time()-t)
        self.save_similarity_batch2(data,
            'mainsite_similarityals',
            1000)
        return solution
