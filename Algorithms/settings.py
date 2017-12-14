"""
                   Settings
===================================================
This file contains settings that allows to tune the system.
"""

"""
===================================================
                GENERAL SETTINGS
===================================================
"""

# List of features to be used to train, test and/or predict movie recomendations
# features_field = [
#         'title_bm25',
#         'genre_bm25',
#         'director_bm25',
#         'writer_bm25',
#         'cast_bm25',
#         'plot_bm25',
#         'full_plot_bm25',
#         'language_bm25',
#         'country_bm25',
#         'awards_bm25',
#         # 'year',
#         # 'als_cosine',
# ]

# # i.e. TFITF
features_field = [
        'title_tfitf',
        'genre_tfitf',
        'director_tfitf',
        'writer_tfitf',
        'cast_tfitf',
        'plot_tfitf',
        'full_plot_tfitf',
        'language_tfitf',
        'country_tfitf',
        'awards_tfitf',
]
#
# # i.e. JACCARD
# features_field = [
#         'title_jaccard',
#         'genre_jaccard',
#         'director_jaccard',
#         'writer_jaccard',
#         'cast_jaccard',
#         'plot_jaccard',
#         'full_plot_jaccard',
#         'language_jaccard',
#         'country_jaccard',
#         'awards_jaccard'
# ]
#
# # i.e. Others
# features_field = [
#         'age_diff',
#         'als_cosine',
#         'weighted_als_cosine',
# ]

# Number of recimmended movies to predict for each movie id input
k = 20

# This id will be used in the generated algorithms table, to identifie temporary recommendations
# which should not be used in the user study
discard_algorithm_id = -1

"""
===================================================
                    TEST SETTINGS
===================================================
"""
# Machine Learning model to use
model = 'lin_reg'
# i.e. Logistic Regression Model
# model = 'log_reg'
# i.e. Support Vector Machine Model
# model = 'svm'

# Flag to decide if the features should be re-generated
# WARNING: Setting this to true can increase testing time by ~10 minutes
generate_features = True

# Flag to decide if the the groundtruth training data should be aggregated
aggregated_training_data = True

# Flag to decided if the model should use standardized coefficients
standardized_coefficients = True


"""
===================================================
                   TRAIN SETTINGS
===================================================
"""
# File system path where the model should be persisted
model_file_path = 'Temp/TMP_MODEL.pkl'


"""
===================================================
                POPULATE SETTINGS
===================================================
"""
# Minimum movie id number
minimum = 0
# Maximum movie id number
maximum = 27300
# Adjust the Number of movies in parallel for which recommendations are generated.
# This parameter only affects performance, and should be adjusted depending on the amount of available memory
steps = 2000
