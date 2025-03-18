#! /usr/bin/env python3

from subprocess import check_output
from glob import glob
import joblib, pickle
import os, io
import torch
torch.use_deterministic_algorithms(True)
import argparse
import numpy as np
import inspect
import time
from copy import deepcopy
import sys

from ase.calculators.calculator import Calculator
from ase.utils import workdir
from ase.io import read,write
import ase.optimize.sciopt

from pymatgen.core import Structure

from .util import Normalizer
from .model import CrystalGraphConvNet



class ASECalculatorGNNs(Calculator):
    """Class for doing a CGCNN energy calculation from a pre-trained model
    """

    implemented_properties = ['energy']
    
    # TODO, if GNN is trained to predict some composition renormalized quantity,
    # provide the dictionary to go back
    def __init__(self,restart=None,ignore_bad_restart_file=False,
                 label='cgcnn', atoms=None, directory='.', renorm_dict=None,
                 **kwargs):

        """ 
        cgcnn_iap : cgcnniap.utils.CGCNN_IAP instance
            The persistent torch model for fast energy calls 
            i.e. avoiding file reading writing, reinitializing model, etc.
        """

        # intitialize super
        Calculator.__init__(self,restart,ignore_bad_restart_file,
                            label, atoms, directory,**kwargs)

        if 'debug' in kwargs.keys() and kwargs['debug']==True:
            self.debug = True
        else:
            self.debug = False
        
        # glob of all kfold models
        modelSearch = os.path.join(kwargs['projdir'], 
                                   kwargs['modelsdir'], 
                                   kwargs['modelsglob'])
        self.modelDirList = sorted(glob(modelSearch))

        
        # IMPORTANT: we are assuming that we are only averaging over K-fold
        # ensemble models, which all have a data feeaturizer that has been identically
        # constructed
        print("=> Loading data featurizer '{}'".format(os.path.join(self.modelDirList[0],'dataset.pth.tar')))
        with open(os.path.join(self.modelDirList[0],'dataset.pth.tar'),'rb') as f:
            self.featurizer = pickle.load(f)
        
        # load the list of k-fold models
        print("Loading models and normalizers...")
        self.modelList = []
        self.normalizerList = []
        for dir_ in self.modelDirList:
            model, normalizer = self.load_model_objects(dir_)
            self.modelList.append(model)
            self.normalizerList.append(normalizer)
            
        # set all models to eval mode
        for model in self.modelList:
            model.eval()

        
    def load_model_objects(self,modelpath):
        
        # super hacky for now, but need the dummy example to get the atom, nbr fea encoding dimensions
        dummy_all_atom_types = [1,1]
        dummy_all_nbrs = [
                          [ # site 0
                            [1, 0.74, 1] # neigh0: [site type, dist, site ind]
                          ],
                          [
                            [1, 0.74, 0]
                          ]
                        ]
        dummyfeatures = self.featurizer.featurize_from_nbr_and_atom_list(dummy_all_atom_types,
                                                                         dummy_all_nbrs,
                                                                         None)
        crystal_atom_idx = [torch.tensor(np.arange(len(dummy_all_atom_types)))]
        orig_atom_fea_len = dummyfeatures[0].shape[-1]
        nbr_fea_len = dummyfeatures[1].shape[-1]
        
        print("=> loading model params '{}'".format(os.path.join(modelpath,'model_best.pth.tar')))
        model_checkpoint = torch.load(os.path.join(modelpath,'model_best.pth.tar'),
                                      map_location=lambda storage, loc: storage)
        model_args = argparse.Namespace(**model_checkpoint['args'])
        print("=> loaded model params '{}'".format(os.path.join(modelpath,'model_best.pth.tar')))


        # Create model
        model = CrystalGraphConvNet(orig_atom_fea_len, nbr_fea_len,                    
                                    atom_fea_len=model_args.atom_fea_len,              
                                    n_conv=model_args.n_conv,                          
                                    h_fea_len=model_args.h_fea_len,                    
                                    n_h=model_args.n_h,                                
                                    classification=True if model_args.task ==          
                                    'classification' else False,                       
                                    Fxyz=True if model_args.task == 'Fxyz' else False,
                                    all_elems=model_args.all_elems,
                                    global_fea_len = 0,
                                    o_fea_len=model_args.o_fea_len,
                                    pooltype=model_args.pooltype)
        model.load_state_dict(model_checkpoint['state_dict'])        
        
        # Normalizer for denorming the model's output
        normalizer = Normalizer(torch.zeros(3))
        normalizer.load_state_dict(model_checkpoint['normalizer'])
        
        return model, normalizer

    def calculate(self,atoms=None, properties=['energy'], system_changes=[],mean_reduce=-1):
        
        # convert ase to to pymatgen
        structure = Structure(atoms.get_cell(),
                              atoms.get_chemical_symbols(),
                              atoms.get_positions(),
                              coords_are_cartesian=True)

        # featurize the cyrstal structure
        
        # atomic identities
        all_atom_types = [structure[i].specie.number for i in range(len(structure))]
        
        # nbr list
        all_nbrs = structure.get_all_neighbors(self.featurizer.radius, include_index=True)
        
        # features built from types and nbr data
        (atom_fea, nbr_fea, nbr_fea_idx, atom_type, nbr_type, nbr_dist, pair_type, 
        nbr_fea_idx_all, gs_fea, gp_fea, gd_fea) = \
            self.featurizer.featurize_from_nbr_and_atom_list(all_atom_types, all_nbrs, None)
        
        # structure site index in batch (trtivial when just one structure)
        crystal_atom_idx = [torch.tensor(np.arange(len(structure)))]
        pool_atom_idx = crystal_atom_idx
        
        # global fea always 0 here
        global_fea = torch.tensor([])

        
        # debug
        # tmp_out = self.modelList[0](
        #     *(atom_fea,
        #     nbr_fea,
        #     nbr_fea_idx,
        #     crystal_atom_idx,
        #     atom_type,
        #     nbr_type,
        #     nbr_dist,
        #     pair_type,
        #     global_fea,
        #     pool_atom_idx)
        # )[:]
        # print(tmp_out)
        # tmp_norm = self.normalizerList[0].denorm(tmp_out[0].data.cpu())[:]
        # print(tmp_norm)

        ener_raw = [model(*(atom_fea,
                            nbr_fea,
                            nbr_fea_idx,
                            crystal_atom_idx,
                            atom_type,
                            nbr_type,
                            nbr_dist,
                            pair_type,
                            global_fea,
                            pool_atom_idx))[:] for model in self.modelList]
        
        #print(ener_raw)
        
        ener_denormed = [norm.denorm(output[0].data.cpu())[:] for norm,output in zip(self.normalizerList,ener_raw)]

        if self.debug:
            #print(ener_raw)
            print(ener_denormed)
        
        #print(ener_denormed)
    
        # feed back raw data across k-folds, don't try to postproess here
        
        #print(ener_ensemble)
        self.results = {'energy': ener_denormed}
        
        return ener_denormed

#if __name__ == "__main__":
#
#    #atoms = read("./data/AlLiMgSnZnHEAOptTraj/AlLiMgSnZn_s233_tag233-5.cif")
#    testfname="/Users/mwitman/Applications/SSHEAGen/AlLiMgSnZn-xml-files/AlLiMgSnZnHEAOptTraj/AlLiMgSnZn_s106_tag106-5.cif"
#    atoms = read(testfname)
#    trueforces=np.loadtxt(testfname[:-4]+"_forces.csv",delimiter=',')
#
#    #atoms = ase.build.make_supercell(atoms,3*np.eye(3))
#
#    calc = CGCNN(command="python3 predict.py model_best.pth.tar data/tmpcalculator/ --resultdir data/tmpcalculator")
#    atoms.set_calculator(calc)
#
#    posoptim = False
#    if posoptim:
#        atomsorig = deepcopy(atoms)
#        atomsorig.set_calculator(calc)
#
#        print(dir(ase.optimize.sciopt))
#
#        print(atoms)
#        opt = ase.optimize.sciopt.SciPyFmin(atoms,trajectory='opt.traj',logfile='opt.log')
#
#        x = opt.run(steps=10)
#        atoms.set_positions(x[0][:-6].reshape(-1,3))
#        atoms.set_cell(x[0][-6:])
#
#
#        finale = atoms.get_potential_energy()
#        orige = atomsorig.get_potential_energy()
#
#        print("orig positions:")
#        print(atomsorig.get_positions())
#        print("final positions:")
#        print(atoms.get_positions())
#
#        print("deltapositions:")
#        print(atoms.get_positions()-atomsorig.get_positions())
#        print("delta UC:")
#        print(atoms.get_cell_lengths_and_angles()-atomsorig.get_cell_lengths_and_angles())
#        print("final E, initial E, delta E:")
#        print(finale, orige, finale-orige)
#
#        write("data/tmpcalculator/tmpopt.cif",atoms)
#
#    numforces = True
#    if numforces:
#
#        #for disp in [0.001,0.01,0.02,0.05,0.1]:
#        for disp in [0.2,0.5,1.0]:
#            start = time.time()
#            forces = atoms._calc.calculate_numerical_forces(atoms,d=disp)
#            end = time.time()
#            elapsed = end-start
#            
#            with open("./testforces/test_numerical_forces_timing.txt","w") as f:
#                f.write("Numerical forces calc time elapsed: %.1f s"%elapsed)
#            
#            np.savetxt("./testforces/test_numerical_forces_disp%f.txt"%disp,forces)
#            np.savetxt("./testforces/true_forces.txt",trueforces)
#            np.savetxt("./testforces/test_numerical_forces_diff_disp%f.txt"%disp,forces-trueforces)
#
#
