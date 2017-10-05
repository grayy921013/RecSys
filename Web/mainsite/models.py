from django.db import models
from django.contrib.auth.models import User


# Create your models here.

class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True)

class Movie(models.Model):
    omdb_id = models.IntegerField(unique=True, db_index=True)
    imdb_id = models.IntegerField(unique=True, db_index=True)
    movielens_id = models.IntegerField(unique=True, db_index=True)
    tmdb_id = models.IntegerField(unique=True, null=True, db_index=True)
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
    popularity = models.FloatField(null=True, db_index=True)
    budget = models.BigIntegerField(null=True)
    revenue = models.BigIntegerField(null=True)
    filtered_plot = models.CharField(max_length=2000,null=True)


class Userinfo(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        primary_key=True,
    )
    age = models.IntegerField(default=0, blank=True)
    gender = models.CharField(max_length=10, default='', blank=True)
    education = models.CharField(max_length=100, default='', blank=True)
    employment = models.CharField(max_length=100, default='', blank=True)
    security_question = models.CharField(max_length=100, default='', blank=True)
    security_answer = models.CharField(max_length=100, default='', blank=True)

    def __unicode__(self):
        return self.user

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
    filtered_plot_tfitf = models.FloatField(null=True)
    filtered_plot_bm25 = models.FloatField(null=True)
    filtered_plot_jaccard = models.FloatField(null=True)

    class Meta:
        unique_together = (('id1', 'id2'),)


class UserVote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie1 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie1")
    movie2 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie2")
    action = models.IntegerField() # -1 for not similar, 0 for skip, 1 for similar



class PasswordReset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

# Temporary Tables #

# we define id1 as the smaller id

class SimilarityPair(models.Model):
    id1_id = models.CharField(max_length=10)
    id2_id = models.CharField(max_length=10)

class SimilarityTitle(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    title_tfitf = models.FloatField(null=True)
    title_bm25 = models.FloatField(null=True)
    title_jaccard = models.FloatField(null=True)
    
class SimilarityGenre(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    genre_tfitf = models.FloatField(null=True)
    genre_bm25 = models.FloatField(null=True)
    genre_jaccard = models.FloatField(null=True)

class SimilarityReleased(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    released_tfitf = models.FloatField(null=True)
    released_bm25 = models.FloatField(null=True)
    released_jaccard = models.FloatField(null=True)

class SimilarityDirector(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    director_tfitf = models.FloatField(null=True)
    director_bm25 = models.FloatField(null=True)
    director_jaccard = models.FloatField(null=True)

class SimilarityWriter(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    writer_tfitf = models.FloatField(null=True)
    writer_bm25 = models.FloatField(null=True)
    writer_jaccard = models.FloatField(null=True)

class SimilarityCast(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    cast_tfitf = models.FloatField(null=True)
    cast_bm25 = models.FloatField(null=True)
    cast_jaccard = models.FloatField(null=True)

class SimilarityMetacritic(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    metacritic_tfitf = models.FloatField(null=True)
    metacritic_bm25 = models.FloatField(null=True)
    metacritic_jaccard = models.FloatField(null=True)

class SimilarityPlot(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    plot_tfitf = models.FloatField(null=True)
    plot_bm25 = models.FloatField(null=True)
    plot_jaccard = models.FloatField(null=True)

class SimilarityFiltered_plot(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    filtered_plot_tfitf = models.FloatField(null=True)
    filtered_plot_bm25 = models.FloatField(null=True)
    filtered_plot_jaccard = models.FloatField(null=True)

class SimilarityFull_plot(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    full_plot_tfitf = models.FloatField(null=True)
    full_plot_bm25 = models.FloatField(null=True)
    full_plot_jaccard = models.FloatField(null=True)

class SimilarityLanguage(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    language_tfitf = models.FloatField(null=True)
    language_bm25 = models.FloatField(null=True)
    language_jaccard = models.FloatField(null=True)

class SimilarityCountry(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    country_tfitf = models.FloatField(null=True)
    country_bm25 = models.FloatField(null=True)
    country_jaccard = models.FloatField(null=True)

class SimilarityAwards(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    awards_tfitf = models.FloatField(null=True)
    awards_bm25 = models.FloatField(null=True)
    awards_jaccard = models.FloatField(null=True)

class SimilarityLast_updated(models.Model):
    id1_id = models.CharField(max_length=10, default='0')
    id2_id = models.CharField(max_length=10, default='0')
    last_updated_tfitf = models.FloatField(null=True)
    last_updated_bm25 = models.FloatField(null=True)
    last_updated_jaccard = models.FloatField(null=True)

class MovieFiltered_Plot(models.Model):
    id = models.CharField(max_length=10, default='0', primary_key=True)
    filtered_plot = models.CharField(max_length=2000,null=True)
