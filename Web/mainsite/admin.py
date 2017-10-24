from django.contrib import admin
from mainsite.models import Userinfo, Movie, Similarity, SimilarMovie, UserVote, SearchAction, VotedMovie, SearchStopword

# Register your models here.
admin.site.register(Userinfo)
admin.site.register(Movie)
admin.site.register(Similarity)
admin.site.register(SimilarMovie)
admin.site.register(UserVote)
admin.site.register(SearchAction)
admin.site.register(VotedMovie)
admin.site.register(SearchStopword)


