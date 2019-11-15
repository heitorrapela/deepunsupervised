#!/bin/sh

METRICS="CE"
TEST_FOLDER="../raw-datasets/Realdata"
PARAM="../arguments/default_som.lhs"
FOLDER="uci_som_250p_b16"

java -jar ClusteringAnalysis.jar ${METRICS} ${TEST_FOLDER} ${FOLDER} . -r 250 -t
