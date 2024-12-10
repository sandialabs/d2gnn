#!/bin/bash

kfold_outer=$1
cvtype=$2
modeltype=$3
flags=$(cat $modeltype/flags.txt)
resdir=$modeltype/$cvtype.k$kfold_outer"_outer"
csvext_train=.$cvtype.k$kfold_outer"_outer.train"
csvext_test=.$cvtype.k$kfold_outer"_outer.test"

echo $kfold_outer $cvtype $modeltype $resdir
echo $flags

if [ ! -d $resdir ]; then
    mkdir -p $resdir;
fi;

cgcnn-defect-train . $flags --csv-ext $csvext_train --resultdir $resdir
