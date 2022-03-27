#!/bin/bash

if [ $# -ne 1 ]
then
    echo "
Usage:
    $0 [outdir [outdir]]
"
    exit 0
fi


OUTDIR=$1
slim -d "OUTDIR='${OUTDIR}'" -s 123 pme.slim &> $OUTDIR/slim.log


python3 plot_density.py $OUTDIR/pme_123.trees
