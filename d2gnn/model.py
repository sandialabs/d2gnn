from __future__ import print_function, division

import torch
import torch.nn as nn
import itertools
import numpy as np
from typing import Tuple,List

from .potentials import energyZBL
#from .data import CIFDataFeaturizer

@torch.jit.script
class CIFDataFeaturizer(object):
    def __init__(self, name:str):
        self.name = name
    def foo(self):
        print("dict:"+self.name)
        ind = [{2:[0,0]}, {1:[1,1]}, {0:[2,2]}]
        # map doesn't seem to be supported
        #print(list(map(lambda x: x[1],ind)))
        

class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    """
    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Initialize ConvLayer.

        Parameters
        ----------

        atom_fea_len: int
          Number of atom hidden features.
        nbr_fea_len: int
          Number of bond features.
        """
        super(ConvLayer, self).__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                                 2*self.atom_fea_len)
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
        self.softplus2 = nn.Softplus()

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Parameters
        ----------

        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom

        Returns
        -------

        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution

        """
        # TODO will there be problems with the index zero padding?
        N, M = nbr_fea_idx.shape
        # convolution
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]
        total_nbr_fea = torch.cat(
            [atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             atom_nbr_fea, nbr_fea], dim=2)
        total_gated_fea = self.fc_full(total_nbr_fea)
        total_gated_fea = self.bn1(total_gated_fea.view(
            -1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
        nbr_sumed = self.bn2(nbr_sumed)
        out = self.softplus2(atom_in_fea + nbr_sumed)
        return out


class CrystalGraphConvNet(nn.Module):
    """
    Create a crystal graph convolutional neural network for predicting total
    material properties.
    """
    def __init__(self, orig_atom_fea_len, nbr_fea_len,
                 atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,
                 classification=False, Fxyz=False, all_elems=[0],
                 global_fea_len=0, o_fea_len=1, pooltype='all',
                 vac_diff_path_len=5):
        """
        Initialize CrystalGraphConvNet.

        Parameters
        ----------

        orig_atom_fea_len: int
          Number of atom features in the input.
        nbr_fea_len: int
          Number of bond features.
        atom_fea_len: int
          Number of hidden atom features in the convolutional layers
        n_conv: int
          Number of convolutional layers
        h_fea_len: int
          Number of hidden features after pooling
        n_h: int
          Number of hidden layers after pooling

        pooltype: str
          Flag that species the pooling type {all, 0}
        

        Fxyz : bool
          Include forces as an additional training target
        """
        super(CrystalGraphConvNet, self).__init__()
        # MW added - due to some torchscripting issues, provide the ability
        # to featurize via model's attributes
        self.dataset1 = CIFDataFeaturizer("name")


        # basic global/defect CGCNN
        self.classification = classification
        self.Fxyz = Fxyz
        self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
        self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                                    nbr_fea_len=nbr_fea_len)
                                    for _ in range(n_conv)])
        self.conv_to_fc = nn.Linear(atom_fea_len+global_fea_len, h_fea_len)
        self.conv_to_fc_softplus = nn.Softplus()
        self.pooltype = pooltype
        self.atom_fea_len = atom_fea_len

        if n_h > 1:
            self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                                      for _ in range(n_h-1)])
            self.softpluses = nn.ModuleList([nn.Softplus()
                                             for _ in range(n_h-1)])

        # architecture mods for defect diffusion
        if 'vac_diff' in self.pooltype:
            self.vac_diff_path_len = vac_diff_path_len
            self.neb = SimpleNEB(atom_fea_len, atom_fea_len, vac_diff_path_len)

            if self.pooltype == 'vac_diff_constrain_2':
                self.conv_decode_img = nn.Linear(atom_fea_len, atom_fea_len)
                self.conv_decode_img_softplus = nn.Softplus()

                self.conv_decode_end = nn.Linear(atom_fea_len, atom_fea_len)
                self.conv_decode_end_softplus = nn.Softplus()

            if self.pooltype == 'vac_diff_constrain_full_split':
                # TODO
                pass
                #self.conv_to_fc_path = nn.Linear(atom_fea_len+global_fea_len, h_fea_len)
                #self.conv_to_fc_softplus_path = nn.Softplus()

                #if n_h > 1:
                #    self.fcs_path = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
                #                              for _ in range(n_h-1)])
                #    self.softpluses_path = nn.ModuleList([nn.Softplus()
                #                                     for _ in range(n_h-1)])
                #self.fc_out_path = nn.Linear(h_fea_len, o_fea_len)

        if self.classification:
            self.fc_out = nn.Linear(h_fea_len, 2)
        else:
            self.fc_out = nn.Linear(h_fea_len, o_fea_len)

        if self.classification:
            self.logsoftmax = nn.LogSoftmax(dim=1)
            self.dropout = nn.Dropout()
        else:
            self.logsoftmax = None
            self.dropout = None

        # MW added - split the network after the convolutions to provide Fxyz 
        # outputs as well
        if self.Fxyz:
            if n_h > 1:
                self.Fxyz_fcs = nn.ModuleList([nn.Linear(atom_fea_len, 
                                                         atom_fea_len)
                                               for _ in range(n_h-1)])
                self.Fxyz_softpluses = nn.ModuleList([nn.Softplus()
                                                 for _ in range(n_h-1)])
            self.conv_to_fc_F = nn.Linear(atom_fea_len,atom_fea_len)
            self.fc_F_out = nn.Linear(atom_fea_len,3)
        else:
            self.Fxyz_fcs = None
            self.Fxyz_softpluses = None
            self.conv_to_fc_F = None
            self.fc_F_out = None

    def forward(self, atom_fea : torch.Tensor, 
                      nbr_fea : torch.Tensor, 
                      nbr_fea_idx : torch.Tensor, 
                      crystal_atom_idx : List[torch.Tensor],
                      atom_type : torch.Tensor, 
                      nbr_type : torch.Tensor, 
                      nbr_dist : torch.Tensor, 
                      pair_type : torch.Tensor,
                      global_fea : torch.Tensor,
                      pool_atom_idx : List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, orig_atom_fea_len)
          Atom features from atom type
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom
        crystal_atom_idx: list of torch.LongTensor of length N0
          len([torch.LongTensor shape (N) , ... ]) == N0
          Mapping from the crystal idx to atom idx in the batch
          e.g. [ LongTensor([0,1]), LongTensor([2,3]), .... ]

        Returns
        -------

        prediction: nn.Variable shape (N, )
          Atom hidden features after convolution ???

        """
        atom_fea = self.embedding(atom_fea)
        #print(atom_fea.shape, "<- embedded atom_fea")
        # >>> torch.Size([N,atom_fea_len])

        for conv_func in self.convs:
            atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
        #print(atom_fea.shape, "<- atom_fea post conv")
        # >>> torch.Size([N,atom_fea_len])

        if self.Fxyz:
            pass
            #crys_Fxyz_out = self.conv_to_fc_F(self.conv_to_fc_softplus(\
            #                                                        atom_fea))
            #crys_Fxyz_out = self.fc_F_out(self.conv_to_fc_softplus(
            #                                                   crys_Fxyz_out))
        #print(crys_Fxyz_out.shape, "<- forces return shape")
        crys_fea = self.pooling(atom_fea, pool_atom_idx, self.pooltype)
        #print(crys_fea.shape, " <- crys_fea post pooling")
        # >>> torch.Size([N0, atom_fea_len])

        # if neb, need to reshape batch so output length is number of energies on the path
        if 'vac_diff' in self.pooltype:

            # reshape to (batch size, seq size, feature size)
            # batch size = num of seq in batch
            # seq size must = num_images + 2 endpoints
            # feature size = dimensionality of site feature vector

            neb_reshape = (len(pool_atom_idx),self.vac_diff_path_len,self.atom_fea_len)
            neb_rereshape = (len(pool_atom_idx)*self.vac_diff_path_len, self.atom_fea_len)

            assert np.allclose(
                #crys_fea[self.vac_diff_path_len].detach().numpy(), 
                #torch.reshape(crys_fea, neb_reshape)[1,0].detach().numpy()
                crys_fea[self.vac_diff_path_len-1].detach().numpy(), 
                torch.reshape(crys_fea, neb_reshape)[0,-1].detach().numpy()
            )
            crys_fea = torch.reshape(crys_fea, neb_reshape)
            #print(crys_fea.shape, " <- crys_fea post vac_diff pooling reshape")

            # Apply NEB seq2seq translation via TransformerEncoder
            neb_fea = self.neb(crys_fea)
            #print(crys_fea.shape, " <- crys_fea post neb")

            if self.pooltype == 'vac_diff':
                crys_fea = neb_fea
            elif self.pooltype == 'vac_diff_constrain':
                crys_fea[:,1:-1,:] = neb_fea[:,1:-1,:]
            elif self.pooltype == 'vac_diff_constrain_1':
                crys_fea = torch.concat(
                    (crys_fea[:,[-1],:], neb_fea[:,1:-1,:], crys_fea[:,[0],:]), axis=1
                )
            elif self.pooltype == 'vac_diff_constrain_2':
                # separately decode sequence end points and images,
                # but compute final energies through same MLP

                # reshapes between batches of sequences (VAC start, VAC end) 
                # to batches of atom features
                end_reshape = (len(pool_atom_idx),2,self.atom_fea_len)
                end_rereshape = (len(pool_atom_idx)*2,self.atom_fea_len)
                
                # extract POSCT CONV sequence of just VACANCY start, end fea vecs
                end_fea = crys_fea[:,[-1,0],:] 
                end_fea = torch.reshape(end_fea, end_rereshape)
                end_fea = self.conv_decode_end(
                    self.conv_decode_end_softplus(end_fea)
                )
                end_fea = torch.reshape(end_fea, end_reshape)

                # reshapes between batches of sequences (IMG 1 ... n)
                img_reshape = (
                    len(pool_atom_idx),
                    (self.vac_diff_path_len-2),
                    self.atom_fea_len
                )
                img_rereshape = (
                    len(pool_atom_idx)*(self.vac_diff_path_len-2),
                    self.atom_fea_len
                )
                
                # extract POST TRANSFORMER ENCODER of just IMG fea vecs 
                img_fea = neb_fea[:,1:-1,:]
                img_fea = torch.reshape(img_fea, img_rereshape)
                img_fea = self.conv_decode_img(
                    self.conv_decode_img_softplus(img_fea)
                )
                img_fea = torch.reshape(img_fea, img_reshape)
                
                # reassemble the neb object
                crys_fea = torch.concat(
                    (end_fea[:,[0],:], img_fea, end_fea[:,[1],:]), axis = 1
                )

            elif self.pooltype == 'vac_diff_constrain_full_split':
                # now we have two MLP decoders which doesn't fit nicely with the single
                # decoder structure for other tasks
                pass
            else:
                raise ValueError('%s not implemented'%self.pooltype)
        

            crys_fea = torch.reshape(crys_fea, neb_rereshape)
            #print(crys_fea.shape, " <- crys_fea post neb reshape")

            # global fea must be repeated for each image in the path    
            global_fea = global_fea.repeat_interleave(self.vac_diff_path_len, dim=0)
            #print(global_fea.shape, " <- global_fea repeated for each image in path")
        
        # post conv / pooling / transformer 
        crys_fea = self.conv_to_fc(\
                    self.conv_to_fc_softplus(torch.cat([crys_fea,global_fea],dim=1))
                   )
        #print(crys_fea.shape, "<- crys_fea conv_to_fc")
        # >>> torch.Size([N0, h_fea_len])

        crys_fea = self.conv_to_fc_softplus(crys_fea)
        #print(crys_fea.shape, "<- activation")
        # >>> torch.Size([N0, h_fea_len])

        #if self.classification:
        #    pass
        #    #crys_fea = self.dropout(crys_fea)
        if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
            for fc, softplus in zip(self.fcs, self.softpluses):
                crys_fea = softplus(fc(crys_fea))
        out = self.fc_out(crys_fea)
        if self.classification:
            pass
            #out = self.logsoftmax(out)
        #print(out.shape, "<- out shape")
        # >>> torch.Size([N0, 1])

        if 'vac_diff' in self.pooltype:
            # reshape output in (number of paths in batch, number of images per path)
            out = out.reshape(len(pool_atom_idx), self.vac_diff_path_len)
            #print(out.shape, "<- out re-shaped for neb path predict")



        # Option 1: Introduce a pair_type, physics-based repulsive potential
        #           requiring a small number of trainable parameter
        #           separate from the GCNN in the computation graph 
        # when optimizing repulsive term parameters, the GCNN output is added
        # to the modeled pairwise repulsive energies to get the total E
        # will need a 
        # pair_type : Variable(torch.Tensor) of shape (N,M)
        #   pair type of each N atoms with their M neighbors 
        #   so tensor has integers (0,1,...) up to (NumElemTypes multichoose 2)
        #   e.g. H-H = 0, H-Mg = 1, Mg-Mg = 2
        #   Note this indexes into the characteristic repulsive term for 
        #   the pair type, as contained in the self.r12coeffs
        # nbr_dist : Variable(torch.Tensor) of shape (N,M)
        #   distances of pairs

        # populate a tensor same size as pair_type with the corresponding 
        # self.r12coeffs
        #target = torch.Tensor([[self.r12coeffs[pair_type[i,j]]\
        #                            for j in range(pair_type.shape[1])]\
        #                       for i in range(pair_type.shape[0])])
        #pw_rep_ener = torch.abs(target)/torch.pow(nbr_dist,12)
        #crys_rep_ener = self.direct_ener_pooling(pw_rep_ener,crystal_atom_idx)
        #print(self.r12coeffs)
        #print(crys_rep_ener.shape, "<- repulsive ener of each crys")
        # >>> torch.Size([N0, 1])


        # Option 2: Introduce a pair_type, physics-based repulsive potential
        #           that does NOT require any fitted parameters
        # Note if atom_type and nbr_types are populated with 0's
        # the ZBL energy will evaluate to zero
        #Zi = torch.unsqueeze(atom_type,dim=1).expand(nbr_type.shape)
        #assert Zi.shape == nbr_type.shape == nbr_dist.shape
        #eZBL = energyZBL(Zi,nbr_type,nbr_dist)
        #crys_rep_ener, crys_size = self.direct_ener_pooling(eZBL,crystal_atom_idx)
       
        ## must divide by two for double counting of all pairs, and normalize
        ## to the per atom total energy 
        #crys_rep_ener = (crys_rep_ener/2)/\
        #                 torch.unsqueeze(torch.tensor(crys_size),dim=1)

        #print('Python output: ')
        #print(out)
        #print(crys_rep_ener)

        if self.Fxyz:
            #raise NotImplemented("No support for forces yet")
            #return out, crys_Fxyz_out
            return [torch.tensor([0]),torch.tensor([0])]
        else:
            #return [torch.add(out,crys_rep_ener), torch.tensor([0])]
            return [out]

    @torch.jit.export
    def compute_repulsive_ener(self, crystal_atom_idx, atom_type, 
                                     nbr_type, nbr_dist):  
        """
        ZBL energy: A physics-based repulsive potential
        that does NOT require any fitted parameters

        Parameters
        ----------

        crystal_atom_idx: list of torch.LongTensor of length N0
          len([torch.LongTensor shape (N) , ... ]) == N0
          Mapping from the crystal idx to atom idx in the batch
          e.g. [ LongTensor([0,1]), LongTensor([2,3]), .... ]
        atom_type : torch.LongTensor shape(N,1)
            Atomic number of each element/node
        nbr_type : torch.LongTensor shape (N,M)
            Atomic number of each neighbor
        nbr_dist : torch.Tensor shape (N,M)
            Neighbor distance

        Returns
        ----------

        crys_rep_ener : torch.Tensor shape (N,)
            The summed ZBL repulsive energy term (pooled) for each crystal 
        """
             


        # Note if atom_type and nbr_types are populated with 0's
        # the ZBL energy will evaluate to zero
        Zi = torch.unsqueeze(atom_type,dim=1).expand(nbr_type.shape)
        assert Zi.shape == nbr_type.shape == nbr_dist.shape
        eZBL = energyZBL(Zi,nbr_type,nbr_dist)
        crys_rep_ener, crys_size = self.direct_ener_pooling(eZBL,crystal_atom_idx)

        # must divide by two for double counting of all pairs, and normalize
        # to the per atom total energy 
        crys_rep_ener = (crys_rep_ener/2)/\
                         torch.unsqueeze(torch.tensor(crys_size),dim=1)

        return crys_rep_ener

    @torch.jit.export
    def direct_ener_pooling(self, pw_ener : torch.Tensor, 
                                  crystal_atom_idx : List[torch.Tensor]):
        # yields repulsive energy of that atomic environment
        atom_ener = torch.sum(pw_ener,dim=1,keepdim=True)
        # sums the repulsive energy for each atomic env in a crystal,
        # across all crystals
        crystal_ener = [torch.sum(atom_ener[idx_map],dim=0,keepdim=True)
                        for idx_map in crystal_atom_idx]
        # crystal_sizes = [len(idx_map) for idx_map in crystal_atom_idx]
        crystal_sizes = [idx_map.shape[0] for idx_map in crystal_atom_idx]
        return torch.cat(crystal_ener, dim=0), crystal_sizes


    @torch.jit.export
    def pooling(self, atom_fea : torch.Tensor, 
                      pool_atom_idx : List[torch.Tensor],
                      pooltype : str):
        """
        Pooling the atom features to crystal features

        N: Total number of atoms in the batch
        N0: Total number of crystals in the batch

        Parameters
        ----------

        atom_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom feature vectors of the batch
        pool_atom_idx: list of torch.LongTensor
          Mapping from the idx in the crystal to idx in batch for pooling
          Must be a list of tensors since each idx_map is 
            tensor of different size (number of atoms in that crystal)
          - length of tensor entry = N0 (num atoms in crystal) if poolind is 'all'
          - length of tensor entry = 1 (single defect atom in crystal) if poolind is 'node'
        pooltype : str
            - 'all' indicates a global pool: avg over all atom features in crystal 
            - 'node' indicates a node pool: extract atom features of a single site
        """
        #assert torch.sum(torch.tensor([len(idx_map) for idx_map in\
        #    crystal_atom_idx])) == atom_fea.data.shape[0]

        #print(atom_fea)
        #print([idx_map for idx_map in pool_atom_idx])
        #print([torch.index_select(atom_fea,0,idx_map)\
        #          for idx_map in pool_atom_idx])


        # 1. average pooling over each index tensor in pool_atom_idx list
        # should be for global feature vectors for  all atoms in each crystal
        if pooltype == 'all':
            pooled_fea = [torch.mean(atom_fea[idx_map], dim=0, keepdim=True)
                          for idx_map in pool_atom_idx]
        # 2. node pools over each tensor indexed by lst entry pool_atom_idx 
        # should be for global feature vectors for  all atoms in each crystal
        elif pooltype == 'node' or 'vac_diff' in pooltype:
            #print([idx_map[0] for idx_map in pool_atom_idx])

            # For defect formation energy, we are really only interested with the feature
            # vector of the node that would become the defect


            # For predicting vacancy diffusion activation energy, model
            # takes nodes corresponding to the diffusing species path, then
            # does a seq2seq transformation via a Transformer encoder
            # which imparts attention to each diffusing node
            pooled_fea = [torch.index_select(atom_fea,0,idx_map)\
                          for idx_map in pool_atom_idx]

            #print(summed_fea)

        #elif pooltype == 'vac_diff':
        #    
        #    print([idx_map for idx_map in pool_atom_idx])
        #    pooled_fea = [torch.index_select(atom_fea,0,idx_map)\
        #                  for idx_map in pool_atom_idx]
    
            
        elif pooltype == 'none':
            return pooled_fea
        else:
            raise ValueError("unallowed pooltype of %s"%pooltype)


        return torch.cat(pooled_fea, dim=0)

class SimpleNEB(nn.Module):
    # https://www.kaggle.com/discussions/general/236840
    # https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html
    # https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html
    
    def __init__(self, dim_fea, dim_feedforward, path_length):
        super(SimpleNEB, self).__init__()
        #self.el1 = nn.TransformerEncoderLayer(d_model=dim_fea, nhead=8, dim_feedforward=dim_feedforward)
        self.el1 = nn.TransformerEncoderLayer(d_model=dim_fea, nhead=8, dim_feedforward=dim_feedforward,batch_first=True)
        self.te1 = nn.TransformerEncoder(self.el1, num_layers=1)
        self.fc1 = nn.Linear(dim_fea,dim_fea)

    def forward(self, x):
        x = self.te1(x)
        x = self.fc1(x)
        return x 
