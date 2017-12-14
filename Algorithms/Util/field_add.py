#-*- coding: utf-8 -*-
"""
               Recomender Engine
===================================================

This script provides the functions to add a new field to mainsite_movie and mainsite_similarity


"""

# Author: Momina Haider. <mominah>

import sys

from DataHandler import PostgresDataHandler


#to add a new field and generate features for that field complete the following steps  after running this script:
#1. Add the Field to the Field Enum
#1.5. Add field to the function get_class_for_field in the Postgres.py file
#2. Add field to default fields array in Main.py
#3. Set the generate_features property to True in the .settings.py file
#4. Run Main.py in test mode
def add(fieldname):
    dataset = PostgresDataHandler()
    #creates a column for the field in the mainsite_movie table. Takes field and datatype as arguments
    # dataset.addMovieField(argv[0],argv[1]);
    #creates a temp table for the field. The table has the provided field with provided datatype as well as id
    # dataset.createFieldTmpTable(argv[0],argv[1]);
    #updates the field in mainsite_movie table from the mainsite_movie<field> table
    # dataset.updateMainsiteMovieField(argv[0]);

    #creates temporary similarity table
    dataset.addSimilarityColumn(fieldname);
    #creates the the columns for the three similarity algorithms in mainsite_similarity table
    dataset.createSimilarity_Field(fieldname);
