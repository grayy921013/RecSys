from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class Genre(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    name = models.CharField(max_length=100)

class Movie(models.Model):
    id = models.CharField(max_length=10, primary_key=True)
    imdb_id = models.CharField(max_length=10, unique=True)
    movielens_id = models.CharField(max_length=10, unique=True)
    tmdb_id = models.CharField(max_length=10, unique=True, null=True)
    title = models.CharField(max_length=500)
    year = models.IntegerField()
    rating = models.FloatField(null=True)
    runtime = models.CharField(max_length=100)
    genres = models.ManyToManyField(Genre)
    released = models.CharField(max_length=2000)
    director = models.CharField(max_length=2000)
    writer = models.CharField(max_length=2000)
    cast = models.CharField(max_length=2000)
    metacritic = models.CharField(max_length=2000)
    imdb_rating = models.FloatField(null=True)
    imdb_votes = models.IntegerField(null=True)
    poster = models.CharField(max_length=1000)
    plot = models.CharField(max_length=2000)
    full_plot = models.CharField(max_length=10000)
    language = models.CharField(max_length=500)
    country = models.CharField(max_length=500)
    awards = models.CharField(max_length=500)
    last_updated = models.CharField(max_length=40)
    popularity = models.FloatField(null=True)
    budget = models.BigIntegerField(null=True)
    revenue = models.BigIntegerField(null=True)

# we define id1 as the smaller id
class Similarity(models.Model):
    id1 = models.ForeignKey(Movie, related_name='id1', on_delete=models.CASCADE)
    id2 = models.ForeignKey(Movie, related_name='id2', on_delete=models.CASCADE)
    title_tfitf = models.FloatField(null=True)
    title_bm25 = models.FloatField(null=True)
    title_jaccard = models.FloatField(null=True)
    runtime_tfitf = models.FloatField(null=True)
    runtime_bm25 = models.FloatField(null=True)
    runtime_jaccard = models.FloatField(null=True)
    genre_tfitf = models.FloatField(null=True)
    genre_bm25 = models.FloatField(null=True)
    genre_jaccard = models.FloatField(null=True)
    released_tfitf = models.FloatField(null=True)
    released_bm25 = models.FloatField(null=True)
    released_jaccard = models.FloatField(null=True)
    director_tfitf = models.FloatField(null=True)
    director_bm25 = models.FloatField(null=True)
    director_jaccard = models.FloatField(null=True)
    writer_tfitf = models.FloatField(null=True)
    writer_bm25 = models.FloatField(null=True)
    writer_jaccard = models.FloatField(null=True)
    cast_tfitf = models.FloatField(null=True)
    cast_bm25 = models.FloatField(null=True)
    cast_jaccard = models.FloatField(null=True)
    metacritic_tfitf = models.FloatField(null=True)
    metacritic_bm25 = models.FloatField(null=True)
    metacritic_jaccard = models.FloatField(null=True)
    plot_tfitf = models.FloatField(null=True)
    plot_bm25 = models.FloatField(null=True)
    plot_jaccard = models.FloatField(null=True)
    full_plot_tfitf = models.FloatField(null=True)
    full_plot_bm25 = models.FloatField(null=True)
    full_plot_jaccard = models.FloatField(null=True)
    language_tfitf = models.FloatField(null=True)
    language_bm25 = models.FloatField(null=True)
    language_jaccard = models.FloatField(null=True)
    country_tfitf = models.FloatField(null=True)
    country_bm25 = models.FloatField(null=True)
    country_jaccard = models.FloatField(null=True)
    awards_tfitf = models.FloatField(null=True)
    awards_bm25 = models.FloatField(null=True)
    awards_jaccard = models.FloatField(null=True)
    last_updated_tfitf = models.FloatField(null=True)
    last_updated_bm25 = models.FloatField(null=True)
    last_updated_jaccard = models.FloatField(null=True)

    class Meta:
        unique_together = (('id1', 'id2'),)


class Userinfo(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    age = models.IntegerField(default=0,blank=True)
    gender = models.CharField(max_length=10, default='',blank=True)
    education = models.CharField(max_length=100, default='',blank=True)
    employment = models.CharField(max_length=100, default='',blank=True)

    def __unicode__(self):
        return self.user

class UserVote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie1 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie1")
    movie2 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie2")
    is_similar = models.BooleanField()
