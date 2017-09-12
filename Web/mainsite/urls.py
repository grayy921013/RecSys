from django.conf.urls import url

from . import views

urlpatterns = [
    url(r'^metadata/(?P<imdb_id>\d+)$', views.get_metadata, name='metadata'),
]