from django.db import models
from django.contrib.auth.models import User


# Create your models here.

# Table for all the genres
class Genre(models.Model):
    name = models.CharField(max_length=100, unique=True)

    def __str__(self):
        return self.name


# Stores all the metadata of movies
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
    writer_processed = models.CharField(max_length=2000, default="")
    cast = models.CharField(max_length=2000)
    metacritic = models.CharField(max_length=2000)
    imdb_rating = models.FloatField(null=True)
    imdb_votes = models.IntegerField(null=True, db_index=True)
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
    tags = models.TextField(default="")

    def __str__(self):
        return self.title + " (" + str(self.year) + ")"


# Stores user information including demographics
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
    has_visited_home = models.BooleanField(default=False)
    has_visited_label = models.BooleanField(default=False)

    def __str__(self):
        return self.user


# Stores pairwise similarity. We define id1 as the smaller id
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
    als_cosine = models.FloatField(null=True)
    libmf_cosine = models.FloatField(null=True)


    class Meta:
        unique_together = (('id1', 'id2'),)

    def __str__(self):
        return "Similarity between " + str(self.id1) + " and " + str(self.id2)


# Stores labels submitted by users
class UserVote(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie1 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie1")
    movie2 = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie2")
    action = models.IntegerField() # -1 for not similar, 0 for skip, 1 for similar
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = (('user', 'movie1', 'movie2'),)

    def __str__(self):
        return str(self.user) + " label of " + str(self.movie1) + " and " + str(self.movie2)


# Stores search actions made by users
class SearchAction(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    keyword = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return "Search " + self.keyword


# Stores tokens generated by password reset
class PasswordReset(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    token = models.CharField(max_length=100, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)


# Stores top N movies recommended by each algorithm for each movie
class SimilarMovie(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="movie")
    similar_movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="similar_movie")
    rank = models.IntegerField() # Rank of each recommended movies. Rank 1 is the most similar.
    algorithm = models.IntegerField()

    def __str__(self):
        return str(self.movie) + " and " + str(self.similar_movie)


# For each user, we store whether he has finished labeling a certain movie
class VotedMovie(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE, related_name="voted_movie")
    finished = models.BooleanField()
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return str(self.user) + " labels on " + str(self.movie)


# Stores words that will be filtered in search
class SearchStopword(models.Model):
    word = models.CharField(max_length=255)

    def __str__(self):
        return self.word


# Stores movies that appear in the ground truth dataset
class GroundTruth(models.Model):
    movielens_id = models.IntegerField(unique=True, db_index=True)

    def __str__(self):
        return str(self.movielens_id)

# Temporary Tables #

# we define id1 as the smaller id

class SimilarityPair(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()

class SimilarityTitle(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    title_tfitf = models.FloatField(null=True)
    title_bm25 = models.FloatField(null=True)
    title_jaccard = models.FloatField(null=True)
    
class SimilarityGenre(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    genre_tfitf = models.FloatField(null=True)
    genre_bm25 = models.FloatField(null=True)
    genre_jaccard = models.FloatField(null=True)

class SimilarityReleased(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    released_tfitf = models.FloatField(null=True)
    released_bm25 = models.FloatField(null=True)
    released_jaccard = models.FloatField(null=True)

class SimilarityDirector(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    director_tfitf = models.FloatField(null=True)
    director_bm25 = models.FloatField(null=True)
    director_jaccard = models.FloatField(null=True)

class SimilarityWriter(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    writer_tfitf = models.FloatField(null=True)
    writer_bm25 = models.FloatField(null=True)
    writer_jaccard = models.FloatField(null=True)

class SimilarityCast(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    cast_tfitf = models.FloatField(null=True)
    cast_bm25 = models.FloatField(null=True)
    cast_jaccard = models.FloatField(null=True)

class SimilarityMetacritic(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    metacritic_tfitf = models.FloatField(null=True)
    metacritic_bm25 = models.FloatField(null=True)
    metacritic_jaccard = models.FloatField(null=True)

class SimilarityPlot(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    plot_tfitf = models.FloatField(null=True)
    plot_bm25 = models.FloatField(null=True)
    plot_jaccard = models.FloatField(null=True)

class SimilarityFiltered_plot(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    filtered_plot_tfitf = models.FloatField(null=True)
    filtered_plot_bm25 = models.FloatField(null=True)
    filtered_plot_jaccard = models.FloatField(null=True)

class SimilarityTags(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    tags_tfitf = models.FloatField(null=True)
    tags_bm25 = models.FloatField(null=True)
    tags_jaccard = models.FloatField(null=True)

class SimilarityFull_plot(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    full_plot_tfitf = models.FloatField(null=True)
    full_plot_bm25 = models.FloatField(null=True)
    full_plot_jaccard = models.FloatField(null=True)

class SimilarityLanguage(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    language_tfitf = models.FloatField(null=True)
    language_bm25 = models.FloatField(null=True)
    language_jaccard = models.FloatField(null=True)

class SimilarityCountry(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    country_tfitf = models.FloatField(null=True)
    country_bm25 = models.FloatField(null=True)
    country_jaccard = models.FloatField(null=True)

class SimilarityAwards(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    awards_tfitf = models.FloatField(null=True)
    awards_bm25 = models.FloatField(null=True)
    awards_jaccard = models.FloatField(null=True)

class SimilarityLast_updated(models.Model):
    id1_id = models.IntegerField()
    id2_id = models.IntegerField()
    last_updated_tfitf = models.FloatField(null=True)
    last_updated_bm25 = models.FloatField(null=True)
    last_updated_jaccard = models.FloatField(null=True)

class MovieFiltered_Plot(models.Model):
    id = models.CharField(max_length=10, default='0', primary_key=True)
    filtered_plot = models.CharField(max_length=2000,null=True)
