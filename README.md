# Defect diffusion graph neural networks (d2gnn)

This project represents modifications and extensions of the original CGCNN code (see below for its README and LICENSE) to predict vacancy defect formation energies and migration energies. Other small modifications have been made to various parts of the code to make some tasks easier: more control over edge features, nested k-fold cross-validation, running on cluster etc.

### Basic usage
Install package using:
```bash
pip install -e .
```
Execute training or prediction tasks using the command line:
```bash
d2gnn-train $flags
d2gnn-predict $flags
```

### Some important CL args
Control target files via CL args to facilitate high-throughput execution across different encoding strategies, cross-validation, etc.
- To change elemenent encoding file
```bash
--init-embed-file $your_atom_init.json
```
- To change the default id_prop.csv file of (structure,property) data to id_prop.csv.your_csv_ext:
```bash
--csv-ext .your_csv_ext
```

### Model prediction types (modifications enabling defect-related predictions)

#### Vacancy formation energies

- Concept: each *symmetrically distinct site* in a crystal structure can have a distinct vacancy formation energy, so a GNN surrogate model must provide a one-to-many prediction, i.e., one structure to multiple vacancy energies. This is accomplished by post-convolution node feature selection corresponding to the vacancy site in question, followed by downstream learning (see https://doi.org/10.1038/s43588-023-00495-2)

- For this application, a complete working example is provided in:
```bash
cd d2gnn/examples/vacancy_formation_energy
./command.sh
```
- The CL argument enabiling the critical functionality is: 
```bash
--pooltype node
```
- The id_prop.csv* file therefore must have the following format:
```bash
<stucture_id_1:tag0>, <index>, <DFT_value>
<stucture_id_1:tag1>, <index>, <DFT_value>
<stucture_id_2:tag0>, <index>, <DFT_value>
<stucture_id_2:tag1>, <index>, <DFT_value>
<stucture_id_2:tag2>, <index>, <DFT_value>
...
```
- The results will be output to test_results.csv (running d2gnn-train) or all_results.csv (running d2gnn-predict):
```bash
<stucture_id_1:tag0>, <index>, <DFT_value>, <model_value>
<stucture_id_1:tag1>, <index>, <DFT_value>, <model_value>
<stucture_id_2:tag0>, <index>, <DFT_value>, <model_value>
<stucture_id_2:tag1>, <index>, <DFT_value>, <model_value>
<stucture_id_2:tag2>, <index>, <DFT_value>, <model_value>
...
```

#### Vacancy migration energies

- Concept: each symmetrically distinct *path* in a crystal structure can have a distinct migration energy, as derived from an NEB calculation. Requirements of a GNN surrogate model become much more complex (see doi: https://doi.org/10.26434/chemrxiv-2024-wrp5z), but it remains a one-to-many prediction, i.e., one structure to multiple migration energies (preferably for all possible migration paths for a given element type within the crystal structure). This is accomplished by introducing virtual nodes along the migration path interpolated along the vector between the start and end sites for the vacancy (i.e., NEB images), assembling the NEB sequence of energies from site and virtual nodes post-convolution, utilizing seq2seq update (e.g., Transformer encoder), and decodeing virtual and site noedes through different output layers to re-assemble the NEB energy sequence.

- For this application, a complete working example is provided in:
```bash
cd d2gnn/examples/vacancy_migration_energy
./command.sh
```
- The CL argument enabiling the critical functionality is: 
```bash
--pooltype vac_diff_constrain_2
```
- The  id_prop.csv* file therefore must have the following format:
```bash
<stucture_id_1:tag0>, <index1>, <index2>, <DFT_value_0>, ... , <DFT_value_n>
<stucture_id_1:tag1>, <index1>, <index2>, <DFT_value_0>, ... , <DFT_value_n>
<stucture_id_2:tag0>, <index1>, <index2>, <DFT_value_0>, ... , <DFT_value_n> 
...
```

#### Global property predictions

- This application recovers the intent of from the original CGCNN code to predict global properties of crystal structures (e.g. formation energy, band gap, etc.)

- The CL argument for this functionality 
```bash
--pooltype all
```
- The  id_prop.csv* file therefore must have the following format:
```bash
<stucture_id_1>, <value>
<stucture_id_2>, <value>
...
```


### Introduction of additional, application-specific local and global featuers

- For a given structure1.cif, can introduce local node attributes (e.g. oxidation state) contained in structure1.cif.locals at the graph encoding stage via:
```bash
--atom-spec locals
``` 
- For a given structure1.cif, can introduce global features (e.g. compound formation enthalpy) contained in structure1.cif.globals at the graph encoding stage via:
```bash
--crys-spec globals
``` 


### How to cite

Please cite the following work if you want to use CGCNN and defect modifications.

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
@article{Witman2023,
  author = {Witman, Matthew D and Goyal, Anuj and Ogitsu, Tadashi and McDaniel, Anthony H. and Lany, Stephan},
  doi = {10.1038/s43588-023-00495-2},
  file = {:Users/mwitman/Research/Papers/WitmanLany{\_}STCH-ML{\_}2023.pdf:pdf},
  issn = {2662-8457},
  journal = {Nat. Comput. Sci.},
  month = {aug},
  number = {8},
  pages = {675--686},
  publisher = {Springer US},
  title = {{Defect graph neural networks for materials discovery in high-temperature clean-energy applications}},
  url = {https://www.nature.com/articles/s43588-023-00495-2},
  volume = {3},
  year = {2023}
}
@misc{Way2024,
  author = {Way, Lauren and Spataru, Catalin and Jones, Reese and Trinkle, Dallas and Rowberg, Andrew and Varley, Joel and Wexler, Robert and Smyth, Christopher and Douglas, Tyra and Bishop, Sean and Fuller, Elliot and McDaniel, Anthony and Lany, Stephan and Witman, Matthew},
  booktitle = {ChemRxiv},
  doi = {10.26434/chemrxiv-2024-wrp5z},
  month = {aug},
  title = {{Defect diffusion graph neural networks for materials discovery in high-temperature, clean energy applications}},
  url = {https://doi.org/10.26434/chemrxiv-2024-wrp5z https://chemrxiv.org/engage/chemrxiv/article-details/66c79806a4e53c487644c72b},
  year = {2024}
}

```

# Crystal Graph Convolutional Neural Networks

This software package implements the Crystal Graph Convolutional Neural Networks (CGCNN) that takes an arbitary crystal structure to predict material properties. 

The package provides two major functions:

- Train a CGCNN model with a customized dataset.
- Predict material properties of new crystals with a pre-trained CGCNN model.

The following paper describes the details of the CGCNN framework:

[Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties](https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)

## Table of Contents

- [How to cite](#how-to-cite)
- [Prerequisites](#prerequisites)
- [Usage](#usage)
  - [Define a customized dataset](#define-a-customized-dataset)
  - [Train a CGCNN model](#train-a-cgcnn-model)
  - [Predict material properties with a pre-trained CGCNN model](#predict-material-properties-with-a-pre-trained-cgcnn-model)
- [Data](#data)
- [Authors](#authors)
- [License](#license)

## How to cite

Please cite the following work if you want to use CGCNN.

```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

##  Prerequisites

This package requires:

- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)

If you are new to Python, the easiest way of installing the prerequisites is via [conda](https://conda.io/docs/index.html). After installing [conda](http://conda.pydata.org/), run the following command to create a new [environment](https://conda.io/docs/user-guide/tasks/manage-environments.html) named `cgcnn` and install all prerequisites:

```bash
conda upgrade conda
conda create -n cgcnn python=3 scikit-learn pytorch torchvision pymatgen -c pytorch -c conda-forge
```

*Note: this code is tested for PyTorch v1.0.0+ and is not compatible with versions below v0.4.0 due to some breaking changes.

This creates a conda environment for running CGCNN. Before using CGCNN, activate the environment by:

```bash
source activate cgcnn
```

Then, in directory `cgcnn`, you can test if all the prerequisites are installed properly by running:

```bash
python main.py -h
python predict.py -h
```

This should display the help messages for `main.py` and `predict.py`. If you find no error messages, it means that the prerequisites are installed properly.

After you finished using CGCNN, exit the environment by:

```bash
source deactivate
```

## Usage

### Define a customized dataset 

To input crystal structures to CGCNN, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `atom_init.json`: a [JSON](https://en.wikipedia.org/wiki/JSON) file that stores the initialization vector for each element. An example of `atom_init.json` is `data/sample-regression/atom_init.json`, which should be good for most applications.

3. `ID.cif`: a [CIF](https://en.wikipedia.org/wiki/Crystallographic_Information_File) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.cif
├── id1.cif
├── ...
```

There are two examples of customized datasets in the repository: `data/sample-regression` for regression and `data/sample-classification` for classification. 

**For advanced PyTorch users**

The above method of creating a customized dataset uses the `CIFData` class in `cgcnn.data`. If you want a more flexible way to input crystal structures, PyTorch has a great [Tutorial](http://pytorch.org/tutorials/beginner/data_loading_tutorial.html#sphx-glr-beginner-data-loading-tutorial-py) for writing your own dataset class.

### Train a CGCNN model

Before training a new CGCNN model, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` to store the structure-property relations of interest.

Then, in directory `cgcnn`, you can train a CGCNN model for your customized dataset by:

```bash
python main.py root_dir
```

You can set the number of training, validation, and test data with labels `--train-size`, `--val-size`, and `--test-size`. Alternatively, you may use the flags `--train-ratio`, `--val-ratio`, `--test-ratio` instead. Note that the ratio flags cannot be used with the size flags simultaneously. For instance, `data/sample-regression` has 10 data points in total. You can train a model by:

```bash
python main.py --train-size 6 --val-size 2 --test-size 2 data/sample-regression
```
or alternatively
```bash
python main.py --train-ratio 0.6 --val-ratio 0.2 --test-ratio 0.2 data/sample-regression
```

You can also train a classification model with label `--task classification`. For instance, you can use `data/sample-classification` by:

```bash
python main.py --task classification --train-size 5 --val-size 2 --test-size 3 data/sample-classification
```

After training, you will get three files in `cgcnn` directory.

- `model_best.pth.tar`: stores the CGCNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the CGCNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained CGCNN model

Before predicting the material properties, you will need to:

- [Define a customized dataset](#define-a-customized-dataset) at `root_dir` for all the crystal structures that you want to predict.
- Obtain a [pre-trained CGCNN model](pre-trained) named `pre-trained.pth.tar`.

Then, in directory `cgcnn`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar root_dir
```

For instace, you can predict the formation energies of the crystals in `data/sample-regression`:

```bash
python predict.py pre-trained/formation-energy-per-atom.pth.tar data/sample-regression
```

And you can also predict if the crystals in `data/sample-classification` are metal (1) or semiconductors (0):

```bash
python predict.py pre-trained/semi-metal-classification.pth.tar data/sample-classification
```

Note that for classification, the predicted values in `test_results.csv` is a probability between 0 and 1 that the crystal can be classified as 1 (metal in the above example).

After predicting, you will get one file in `cgcnn` directory:

- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set. Here the target value is just any number that you set while defining the dataset in `id_prop.csv`, which is not important.

## Data

To reproduce our paper, you can download the corresponding datasets following the [instruction](data/material-data).

## Authors

This software was primarily written by [Tian Xie](http://txie.me) who was advised by [Prof. Jeffrey Grossman](https://dmse.mit.edu/faculty/profile/grossman). 

## License

CGCNN is released under the MIT License.



