import os
import json
from DataHandler.Postgres import PostgresDataHandler


def add_tmdb():
    dataset = PostgresDataHandler()
    dataset.createTempTMDBTable()
    parse_tmdb_json()
    dataset.changeTMDB()
    dataset.updateTMDB_similarMovie()



def parse_tmdb_json(batch = 2000):
    #create the temp_tmdb table
    dataset = PostgresDataHandler()
    tuples = []
    with open('./Data/TMDBTotal.json') as fileobject:
        no_record =0
        for line in fileobject:
            line_object = json.loads(line)
            similar_move_index = "similar_movies top "
            print(line_object)
            for i in range(20):
                try:
                    tuples.append((line_object["id"],line_object[similar_move_index + str(i)],str(i)))
                except Exception as e:
                    continue


            no_record = no_record + 1
            if no_record == batch:
                dataset.insertTMDB_Json(tuples)
                tuples =[]
                no_record = 0












