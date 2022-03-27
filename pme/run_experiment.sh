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
slim -d "OUTDIR=\'${OUTDIR}\'" pme.slim &> $OUTDIR/slim.log


