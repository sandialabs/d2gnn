#! /bin/bash

d2gnn-train . $(cat flags.txt) --csv-ext .train_k1 --resultdir models

d2gnn-predict models/model_best.pth.tar . -CIFdatapath models/dataset.pth.tar --resultdir models --csv-ext .hold_k1 --disable-cuda
