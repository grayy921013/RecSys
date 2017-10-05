#!/bin/bash
FILES=/Users/Momina/Documents/CMU_SV_Fall_2017/Practicuum/Stanford-NER/stanford-ner-2017-06-09/test_dir/*
for f in $FILES
do
	java -mx600m  edu.stanford.nlp.ie.crf.CRFClassifier -loadClassifier classifiers/english.all.3class.distsim.crf.ser.gz -outputFormat tabbedEntities -textFile "$f"  > "${f}_result.tsv"
done
