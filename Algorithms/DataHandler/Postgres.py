from DataHandler.Base import DataHandler
from Util import Field
from peewee import PostgresqlDatabase, Model, CharField, IntegerField, FloatField
import os

# TODO: Move this to be passed as parameters
db = PostgresqlDatabase(
    'recsys',  # Required by Peewee.
    user=os.environ['POSTGRES_USER'],  # Will be passed directly to psycopg2.
    password=os.environ['POSTGRES_PASSWORD'],  # Ditto.
    host='54.193.113.161',  # Ditto.
)

import logging


# Create your models here.
class mainsite_movie(Model):
    id = CharField(max_length=10, primary_key=True)
    imdb_id = CharField(max_length=10, unique=True)
    movielens_id = CharField(max_length=10, unique=True)
    title = CharField(max_length=500)
    year = IntegerField()
    rating = FloatField(null=True)
    runtime = CharField(max_length=100)
    genre = CharField(max_length=2000)
    released = CharField(max_length=2000)
    director = CharField(max_length=2000)
    writer = CharField(max_length=2000)
    cast = CharField(max_length=2000)
    metacritic = CharField(max_length=2000)
    imdb_rating = FloatField(null=True)
    imdb_votes = IntegerField(null=True)
    poster = CharField(max_length=1000)
    plot = CharField(max_length=2000)
    full_plot = CharField(max_length=10000)
    language = CharField(max_length=500)
    country = CharField(max_length=500)
    awards = CharField(max_length=500)
    last_updated = CharField(max_length=40)

    class Meta:
        database = db  # This model uses the "people.db" database.

class PostgresDataHandler(DataHandler):
    def __init__(self):
        db.connect()

    def get_data(self, field):
        if not isinstance(field, Field):
            raise AttributeError("Parameter field should be value of the enum Field")

        if field == Field.TITLE:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.title)
            return list(map(lambda x: [x.movielens_id, x.title], cursor))
        elif field == Field.GENRE:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.genre)
            return list(map(lambda x: [x.movielens_id, x.genre], cursor))
        elif field == Field.CAST:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.cast)
            return list(map(lambda x: [x.movielens_id, x.cast], cursor))
        elif field == Field.PLOT:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.plot)
            return list(map(lambda x: [x.movielens_id, x.plot], cursor))
        elif field == Field.FULL_PLOT:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.full_plot)
            return list(map(lambda x: [x.movielens_id, x.full_plot], cursor))
        elif field == Field.AWARDS:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.awards)
            return list(map(lambda x: [x.movielens_id, x.awards], cursor))

        elif field == Field.DIRECTOR:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.director)
            return list(map(lambda x: [x.movielens_id, x.director], cursor))
        elif field == Field.WRITER:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.writer)
            return list(map(lambda x: [x.movielens_id, x.writer], cursor))
        elif field == Field.LANGUAGE:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.language)
            return list(map(lambda x: [x.movielens_id, x.language], cursor))
        elif field == Field.COUNTRY:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.country)
            return list(map(lambda x: [x.movielens_id, x.country], cursor))

        elif field == Field.YEAR:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.year)
            return list(map(lambda x: [x.movielens_id, x.year], cursor))
        elif field == Field.RATING:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.rating)
            return list(map(lambda x: [x.movielens_id, x.rating or ''], cursor))
        elif field == Field.RUNTIME:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.runtime)
            return list(map(lambda x: [x.movielens_id, x.runtime], cursor))
        elif field == Field.RELEASED:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.released)
            return list(map(lambda x: [x.movielens_id, x.released], cursor))
        elif field == Field.METACRITIC:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.metacritic)
            return list(map(lambda x: [x.movielens_id, x.metacritic], cursor))
        elif field == Field.IMDB_RATING:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.imdb_rating)
            return list(map(lambda x: [x.movielens_id, x.imdb_rating], cursor))
        elif field == Field.IMDB_VOTES:
            cursor = mainsite_movie.select(mainsite_movie.movielens_id, mainsite_movie.imdb_votes)
            return list(map(lambda x: [x.movielens_id, x.imdb_votes], cursor))
        else:
            return []

    def __del__(self):
        db.close()
