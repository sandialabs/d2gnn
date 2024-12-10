#!/bin/bash

kfold_outer=$1
cvtype=$2
modeltype=$3
flags=$(cat $modeltype/flags.txt)
resdir=$modeltype"/"$cvtype".k"$kfold_outer"_outer"
csvext_test="."$cvtype".k"$kfold_outer"_outer.test"

echo $kfold_outer $cvtype $modeltype $resdir
echo $flags

cgcnn-defect-predict $resdir/model_best.pth.tar . -CIFdatapath $resdir/dataset.pth.tar --resultdir $resdir --csv-ext $csvext_test --disable-cuda
