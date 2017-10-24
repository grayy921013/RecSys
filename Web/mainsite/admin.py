from django.contrib import admin
from mainsite.models import Userinfo, Movie, Similarity, SimilarMovie, UserVote, SearchAction, VotedMovie, SearchStopword


class SearchActionAdmin(admin.ModelAdmin):
    readonly_fields = ('created_at', 'user')


class UserVoteAdmin(admin.ModelAdmin):
    readonly_fields = ('updated_at', 'movie1', 'movie2', 'user', 'label')
    exclude = ('action',)

    def label(self, obj):
        if obj.action == -1:
            return "Not similar"
        elif obj.action == 0:
            return "Skip"
        else:
            return "Similar"


class VotedMovieAdmin(admin.ModelAdmin):
    readonly_fields = ('updated_at', 'user')


class SimilarMovieAdmin(admin.ModelAdmin):
    readonly_fields = ('movie', 'similar_movie')


class SimilarityAdmin(admin.ModelAdmin):
    readonly_fields = ('id1', 'id2')

# Register your models here.
admin.site.register(Userinfo)
admin.site.register(Movie)
admin.site.register(Similarity, SimilarMovieAdmin)
admin.site.register(SimilarMovie, SimilarMovieAdmin)
admin.site.register(UserVote, UserVoteAdmin)
admin.site.register(SearchAction, SearchActionAdmin)
admin.site.register(VotedMovie, VotedMovieAdmin)
admin.site.register(SearchStopword)


