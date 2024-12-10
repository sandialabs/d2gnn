#! /bin/bash

d2gnn-train . $(cat flags.txt) --csv-ext .0.k0_outer.train --resultdir models

d2gnn-predict models/model_best.pth.tar . -CIFdatapath models/dataset.pth.tar --resultdir models --csv-ext .0.k0_outer.train --disable-cuda
