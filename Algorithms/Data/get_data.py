# coding: utf-8

# In[45]:

import re
import pandas as pd
import numpy as np
import json
import logging

logger = logging.getLogger(__name__)

# In[47]:

# Reading Movies
def get_data():
    movies = pd.read_csv(r'Data\ml-20m\movies.csv', sep=',').values

    size = len(movies)
    logger.debug('Read %d rows', size)

    c_id_field = [None] * size
    c_title_field = [None] * size
    c_year_field = [None] * size
    c_genres_field = [None] * size

    i = 0
    for movie in movies:
        title = movie[1]
        match = re.match('(.+)(\()(\d\d\d\d)(\))', title)
        if match:
            c_id_field[i] = movie[0]
            c_title_field[i] = match.group(1)
            c_year_field[i] = match.group(3)
            c_genres_field[i] = movie[2].replace('|', ' ')
            i += 1
        else:
            pass
    logger.debug("Processed %d rows", i)

    c_id_field = c_id_field[:i]
    c_title_field = c_title_field[:i]
    c_year_field = c_year_field[:i]
    c_genres_field = c_genres_field[:i]

    return c_id_field, c_title_field, c_year_field, c_genres_field


# In[48]:

# Reading Tags
def get_tags(c_id_field):
    df_tags = pd.read_csv(r'Data\ml-20m\tags.csv', sep=',', encoding='utf-8')

    logger.debug('Read %d rows', len(df_tags))

    c_tags_field = [None] * len(c_id_field)

    i = 0
    j = 0
    size = len(c_id_field)
    for movie_id in c_id_field:
        i += 1
        tags_of_movie = df_tags[df_tags['movieId'] == movie_id].values

        string = ''
        for tag in tags_of_movie:
            if isinstance(tag[2], basestring):
                string += tag[2].replace(' ', '_') + ' '
            else:
                pass
                # print tag

        if string != '':
            j += 1

        c_tags_field[i - 1] = string


    return c_tags_field


import numpy as np
import numbers


def from_tmdb_to_movielens_id(links, tmdbid):
    if not tmdbid in links.index:
        return -1

    link = links.loc[tmdbid]
    if link is None:
        return -1

    movie_id = link['movieId']

    if not isinstance(movie_id, numbers.Number):
        # TODO: Handle tmdbid collisions
        return -1

    return movie_id


def from_tmdb_to_movielens_id2(links, ids, tmdbid):
    if not tmdbid in links.index:
        return -1

    link = links.loc[tmdbid]
    if link is None:
        return -1

    movie_id = link['movieId']

    if not isinstance(movie_id, numbers.Number):
        # TODO: Handle tmdbid collisions
        # print 'hey'
        return -1

    # print tmdbid, '...', link, '>>>', movie_id
    if not movie_id in ids:
        return -1

    return movie_id


import os.path
import pickle

def get_related_tmdb(rank_length):
    if os.path.isfile(r'Data\tmdb.pkl'):
        pkl_file = open(r'Data\tmdb.pkl', 'rb')
        data1 = pickle.load(pkl_file)
        pkl_file.close()
        return data1

    links = pd.read_csv(r'Data\ml-20m\links.csv', sep=',')
    # Drop links where we dont have a tmdbid referral
    links = links[np.isfinite(links['tmdbId'])]
    # Finally use tmdbid as index
    links = links.set_index('tmdbId')

    size = len(links)
    logger.info('Total links %d' % (size))

    file = open(r'Data\TMDBTotal.json')

    output = []
    ids = []
    c = 0
    for line in file:
        if c % 1000 == 0:
            logger.warn("Processing line: %d" % c)
        c += 1

        match = re.match(r'.*"id": (\d+),"imdb_id".*', line)
        if not match:
            continue

        tmdbid = int(match.group(1))

        movie_id = from_tmdb_to_movielens_id(links, tmdbid)

        if movie_id < 0:
            logger.info('Skipped %s' % match.group(1))
            continue

        movie = json.loads(line)
        rec = set()
        for i in range(min(rank_length,20)):
            if ('similar_movies top %d' % i) in movie:
                tmdb_id = movie['similar_movies top %d' % i]
                movielens_id = from_tmdb_to_movielens_id(links, tmdb_id)
                if movielens_id > 0:
                    rec.add(movielens_id)
            else:
                break
        output.append(rec)
        ids.append(movie_id)

    file.close()
    logger.info('Processed %d' % len(output))

    result = (output, ids)

    # Pickle the list using the highest protocol available.
    output = open(r'Data\tmdb.pkl', 'wb')
    pickle.dump(result, output, -1)
    output.close()

    return result

    # related_movies = output
    #
    # size_ids = len(c_id_field)
    # related_movies_set = [None] * size_ids
    # idx = 0
    # j = 0
    #
    # for i in c_id_field:
    #     i = int(i)
    #     if i in related_movies:
    #         related_movies_set[idx] = related_movies[i]
    #     else:
    #         related_movies_set[idx] = set()
    #         #         print 'Skipped', i
    #         j += 1
    #     idx += 1
    #
    # logger.info('Movies %d Skipped %d' % (size_ids, j))
    # return related_movies_set

def get_related_tmdb2(c_id_field):
    links = pd.read_csv(r'Data\ml-20m\links.csv', sep=',')
    # Only use links that we have more data for
    links = links[links['movieId'].isin(c_id_field)]
    # Drop links where we dont have a tmdbid referral
    links = links[np.isfinite(links['tmdbId'])]
    # Finally use tmdbid as index
    links = links.set_index('tmdbId')

    size = len(links)
    logger.info('Total Movies %d Total links %d' % (len(c_id_field), size))
    # print links[0:10]
    # print links.loc[862]

    file = open(r'Data\TMDBTotal.json')

    output = {}

    for line in file:
        match = re.match(r'.*"id": (\d+),"imdb_id".*', line)
        if not match:
            continue

        tmdbid = int(match.group(1))

        movie_id = from_tmdb_to_movielens_id(links, c_id_field, tmdbid)

        if movie_id < 0:
            logger.info('Skipped %s' % match.group(1))
            continue

        movie = json.loads(line)
        rec = set()
        for i in range(20):
            if ('similar_movies top %d' % i) in movie:
                tmdb_id = movie['similar_movies top %d' % i]
                movielens_id = from_tmdb_to_movielens_id(links, c_id_field, tmdb_id)
                if movielens_id > 0:
                    rec.add(movielens_id)
            else:
                break
        output[movie_id] = rec

    file.close()
    logger.info('Processed %d' % len(output))

    #return output
    related_movies = output

    size_ids = len(c_id_field)
    related_movies_set = [None] * size_ids
    idx = 0
    j = 0

    for i in c_id_field:
        i = int(i)
        if i in related_movies:
            related_movies_set[idx] = related_movies[i]
        else:
            related_movies_set[idx] = set()
            j += 1
        idx += 1

    logger.info('Movies %d Skipped %d' % (size_ids, j))
    return related_movies_set

def get_related(filename):
    data = pd.read_csv(filename, sep=',')
    data = data.dropna()
    data = data.values

    result = set()

    for record in data:
        # We make sure that the minimum id is always first to eliminate duplicates
        # Ex. A record for 1,2 and another for 2,1
        small = min(record[0], record[1])
        big = max(record[0], record[1])
        result.add((small, big, record[2] == 1))

    result = list(map(lambda x: list(x), result))
    return result
