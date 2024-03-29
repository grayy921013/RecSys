from django.conf.urls import url
from . import views

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^refresh$', views.refresh, name='refresh'),
    url(r'^blockbuster$', views.blockbuster, name='blockbuster'),
    url(r'^metadata/(?P<imdb_id>\d+)$', views.get_metadata, name='metadata'),
    url(r'^label/(?P<id>\d+)$', views.label, name='label'),
    url(r'^userlogin$', views.userlogin, name='userlogin'),
    url(r'^userlogout$', views.userlogout, name='userlogout'),
    url(r'^profile$', views.profile, name='profile'),
    url(r'^register$', views.register, name='register'),
    url(r'^resetpwd', views.reset_pwd, name='resetpwd'),
    url(r'^modifypwd/(?P<uid>\d+)/(?P<token>.+)', views.modify_pwd, name='modifypwd'),
    url(r'^search$', views.search, name='search'),
    url(r'^getsimilar/(?P<id>\d+)$', views.get_similar_movies, name='getsimilar'),
    url(r'^uservote$', views.user_vote, name='uservote'),
    url(r'^check_visited$', views.check_visited, name='check_visited'),
    url(r'^visit_page$', views.visit_page, name='visit_page'),
    url(r'^get_vote_count$', views.get_vote_count, name='get_vote_count')
]