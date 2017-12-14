# Recommender Engine
This module implements recommendations algorithms:
- CB (Content Based)
    - TFIDF
    - BM25
    - JACCARD
- CF (Collaborative Filtering)
    - Latent Factors -> ALS

It allows persisting features and recommendations using DataHandlers:
- PostgreSQL

Finally, it combines the previously generated features using one of the machine learning models:
- Linear Regression
- Logistic Regression
- Support Vector Machine

# Usage
Usage: `python Main.py [-h] [-algorithm [Algorithm ID]] [-movie [Movie ID]] command [file_path]`


| Command | Description |
| --- | --- |
| test | Use a ground truth file to evaluate effectiveness of some feature combination. Set the features in the settings.py file |
| train | Use a complete ground truth file to train a machine learning model and persist it to the file system |
| als | Train a Matrix Factorization Model and generate cosine similarity features between the movies vectors of the train model, using the library MyMediaLite
| als_libmf | Train a Matrix Factorization Model and generate cosine similarity features between the movies vectors of the train model, using the library LIBMF. |
| populate | Calculate and persist the best 20 movie recommendations for each of the movies in the database |
| predict | Given a movie id it predicts the best 20 movie recommendations for that movie |
| add_tmdb | Aggregates the recommendations of the tmdb website as an extra algorithm to the database. |
| age_diff | Populate the age_diff feature in the features table |

# Nice To Have

[] Perform the truncate and sorting of movies for CB Algorithms in a batched way, to allow scaling of data, while being able to run in machines with limited RAM memory.
[] Allow to specify in the settings file which: algorithms, fields and movies should be considered when generating features.