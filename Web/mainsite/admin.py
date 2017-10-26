from django.contrib import admin
from mainsite.models import Userinfo, Movie, Similarity, SimilarMovie, UserVote, SearchAction, VotedMovie, SearchStopword
from django.conf.urls import url
from django.shortcuts import render
from django.db import connection

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

    def get_urls(self):
        urls = super().get_urls()
        my_urls = [
            url('analytics/', self.get_analytics),
        ]
        return my_urls + urls

    def get_analytics(self, request):
        context = dict()
        not_similar_count = {}
        precision = {}
        with connection.cursor() as cursor:
            cursor.execute("select count(CASE WHEN action=-1 THEN 1 END), algorithm from " +
                           "mainsite_similarmovie,mainsite_uservote where movie1_id=movie_id and " +
                           "movie2_id=similar_movie_id group by algorithm;")
            not_similar = cursor.fetchall()
            for tuple in not_similar:
                not_similar_count[tuple[1]] = tuple[0]
            cursor.execute("select count(CASE WHEN action=1 THEN 1 END), algorithm from " +
                           "mainsite_similarmovie,mainsite_uservote where movie1_id=movie_id and " +
                           "movie2_id=similar_movie_id group by algorithm;")
            similar = cursor.fetchall()
            for tuple in similar:
                precision[tuple[1]] = tuple[0] / (tuple[0] + not_similar_count[tuple[1]])
            context['precision'] = precision

        return render(request, 'analytics.html', context)


class VotedMovieAdmin(admin.ModelAdmin):
    readonly_fields = ('updated_at', 'user')


class SimilarMovieAdmin(admin.ModelAdmin):
    readonly_fields = ('movie', 'similar_movie')


class SimilarityAdmin(admin.ModelAdmin):
    readonly_fields = ('id1', 'id2')

# Register your models here.
admin.site.register(Userinfo)
admin.site.register(Movie)
admin.site.register(Similarity, SimilarityAdmin)
admin.site.register(SimilarMovie, SimilarMovieAdmin)
admin.site.register(UserVote, UserVoteAdmin)
admin.site.register(SearchAction, SearchActionAdmin)
admin.site.register(VotedMovie, VotedMovieAdmin)
admin.site.register(SearchStopword)


