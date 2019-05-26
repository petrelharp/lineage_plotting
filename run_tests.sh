#!/bin/bash

set -eu

SEED=$RANDOM
OUTBASE="flat_map/run_BURNIN_2_seed_${SEED}"

./animate_individuals.py 10 flat_map.slim seed=${SEED} BURNIN=2
OUTFILE=${OUTBASE}.pop.mp4
if [ ! -f $OUTFILE ]
then
    echo "FAILED:"
    echo "  ./animate_individuals.py 10 flat_map.slim seed=${SEED} BURNIN=2"
    exit 1
else
    rm $OUTFILE
fi

./animate_trios.py 10 flat_map.slim seed=${SEED} BURNIN=2
OUTFILE=${OUTBASE}.trios.mp4
if [ ! -f $OUTFILE ]
then
    echo "FAILED:"
    echo "  ./animate_trios.py 10 flat_map.slim seed=${SEED} BURNIN=2"
    exit 1
else
    rm $OUTFILE
fi

./animate_ancestry_tree.py 10 3 flat_map.slim seed=${SEED} BURNIN=2
OUTFILE=${OUTBASE}.trees.mp4
if [ ! -f $OUTFILE ]
then
    echo "FAILED:"
    echo "  ./animate_ancestry_tree.py 10 3 flat_map.slim seed=${SEED} BURNIN=2"
    exit 1
else
    rm $OUTFILE
fi

./animate_lineage_tree.py 10 3 2 flat_map.slim seed=${SEED} BURNIN=2
OUTFILE=${OUTBASE}.lineages.mp4
if [ ! -f $OUTFILE ]
then
    echo "FAILED:"
    echo "  ./animate_lineage_tree.py 10 3 2 flat_map.slim seed=${SEED} BURNIN=2"
    exit 1
else
    rm $OUTFILE
fi

rm ${OUTBASE}.trees
rm ${OUTBASE}.log
