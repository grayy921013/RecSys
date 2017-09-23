from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^$', views.home, name='home'),
    url(r'^blockbuster$', views.blockbuster, name='blockbuster'),
    url(r'^metadata/(?P<imdb_id>\d+)$', views.get_metadata, name='metadata'),
    url(r'^userlogin$', views.userlogin, name='userlogin'),
    url(r'^userlogout$', views.userlogout, name='userlogout'),
    url(r'^profile/(?P<id>\d+)$', views.profile, name='profile'),
    url(r'^register$', views.register, name='register')
]