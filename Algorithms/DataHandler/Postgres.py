from DataHandler.Base import DataHandler
from Util import Field
import os
import sys
sys.path.append('../')
sys.path.append('../Web')
os.environ['DJANGO_SETTINGS_MODULE']='Web.settings'
import django
django.setup()
from mainsite.models import Movie, Similarity

import logging

class PostgresDataHandler(DataHandler):
    def __init__(self):
        self.batch = []
        self.cache = {}

    def clear_similarity(self):
        similarities = Similarity.objects.all()
        similarities.delete()

    def add_to_similarity_batch(self, id1, id2, field, algo, score):

        if (id1, id2) in self.cache:
            s = self.batch[self.cache[(id1, id2)]]
            flag = True
        else:
            s = Similarity(
                id1_id=id1, id2_id=id2
            )
            flag = False
            self.cache[(id1, id2)] = len(self.batch)

        field_name = str(field).split('.')[1].lower()
        db_fieldname = ("%s_%s" % (field_name, algo)).lower()
        if not hasattr(s, db_fieldname):
            raise Exception("This field is not in the DB: " + db_fieldname)
        setattr(s, db_fieldname, score)
        
        if not flag:
            self.batch.append(s)


    def save_similarity_batch(self):
        # print len(self.batch)
        Similarity.objects.bulk_create(self.batch)
        self.batch = []

        data = Similarity.objects.all()[:10]
        for a in data:
            print a.title_tfitf, a.title_bm25, a.title_jaccard

    def save_similarity(self, id1, id2, field, algo, score):
        # TODO: Move this to another function
        field_name = str(field).split('.')[1].lower()
        db_fieldname = ("%s_%s" % (field_name, algo)).lower()
        return Similarity.objects.update_or_create(
            id1_id=id1, id2_id=id2, defaults={db_fieldname: score}
        )

    def get_data(self, field, movielens_ids = []):
        if not isinstance(field, Field):
            raise AttributeError("Parameter field should be value of the enum Field")

        if str(field) == 'Field.GENRE':
            field_name = 'genres'
        else:
            field_name = str(field).split('.')[1].lower()

        if len(movielens_ids):
            cursor = Movie.objects.filter(movielens_id__in=movielens_ids).values_list('id', field_name)
        else:
            cursor = Movie.objects.all().values_list('id', field_name)
        
        return list(map(lambda x: [x[0], x[1] or ''], cursor))
