#-*- coding: utf-8 -*-
"""
               Recomender Engine
===================================================

This module implements recommendations algorithms:
    - CB (Content Based)
        - TFIDF
        - BM25
        - JACCARD

"""

# Author: Momina Haider. <mominah>

import sys
import logging
import io
import numpy as np
import nltk
from time import time
from nltk.tag.stanford import StanfordNERTagger
#~~~                                    Do a basic silly run                                 ~~~#
from DataHandler import PostgresDataHandler
from Util import Field
from mainsite.models import Movie,MovieFiltered_Plot





def main(argv):
    snert = StanfordNERTagger('nerTools/Stanford-NER/stanford-ner-2017-06-09/classifiers/english.all.3class.distsim.crf.ser.gz', 'nerTools/Stanford-NER/stanford-ner-2017-06-09/stanford-ner.jar')
    logging.basicConfig(level=logging.WARN)
    # Get the connection with the 'database'
    dataset = PostgresDataHandler()
    # Single Field
    data = dataset.get_data(Field.PLOT)
    matrix = np.array(data)
    ids = matrix[:, 0].astype(int)
    values = matrix[:, 1]
    movie_index = 0
    bulk = []
    bulk_count = 0
    t = time()
    print(t)
    for value in values:
        print ids[movie_index]
        tokens = nltk.word_tokenize(value.encode('utf-8').decode('utf-8'))
        tagged = snert.tag(tokens)
        #ne = nltk.ne_chunk(tagged_tokens=tagged,binary=True)
        sentence = ""
        for word in tagged:
            if word[1] != u'PERSON':
                sentence = sentence + " " + word[0]
        sentence = sentence.lstrip()
        bulk.append(MovieFiltered_Plot(id = ids[movie_index], filtered_plot=sentence))
        bulk_count = bulk_count + 1
        if bulk_count >= 1000:
            print(time() - t)
            print("Saved Bulk")
            t = time()
            MovieFiltered_Plot.objects.bulk_create(bulk)
            bulk_count = 0
            bulk = []
        #movie = Movie.objects.get(id=ids[movie_index])
        #movie.filtered_plot = sentence
        #movie.save()
        movie_index = movie_index + 1
    if bulk.size:
        MovieFiltered_Plot.objects.bulk_create(bulk)


if __name__ == "__main__":
    main(sys.argv)