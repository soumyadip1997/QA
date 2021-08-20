import torch
import glob
import pandas as pd
from neigh import neigh1
from one_hot_atom import atom1
import numpy as np
from one_hot_res import res1
from torch.utils.data import Dataset
from torch.utils.data import sampler

from Bio.PDB import *

import warnings

class protein_read(Dataset):

    def __init__(self, train_location,train_labels):
        #train_data_files_list contain the list of all decoy  pdb files used for training
        self.train_data_files_list = glob.glob(train_location)
        #train_labels contain the location of the all the .txt  files that contains the GDT-TS information
        
        self.train_labels=train_labels
    def __len__(self):
        return len(self.train_data_files_list)

    def __getitem__(self, loc1):
           # try:
                #reading the gdtts scores from the .txt file
                #taking the first pdb file from the list of pdb files
                train_data_files=self.train_data_files_list[loc1]
                #getting the target name  as the files are located in the following pattern d/Data/T1051/T1051_TS006 where T1051 is a target and T1051_TS006 is a decoy
                target_name=train_data_files.split("/")[2]
                #getting the decoy name
                decoy_name=train_data_files.split("/")[3]
                #getting the path of the train labels for that target
                train_label=self.train_labels+"/"+target_name+".txt"
                #reading the file that contains the GDt-TS information of tha target for all the decoys
                A12=pd.read_fwf(train_label)
                
                c111=np.array(A12.columns)
                target=np.array(A12[c111[3]],dtype=np.float)
                target=target[1:-1]
                target=target/100
                weights=target
                decoy_list=np.array(A12[c111[1]],dtype=np.str)
                decoy_list=decoy_list[1:-1]
                #finding out the GDT-TS score for the decoy that we  currently selected from all the decoys of that target
                required_gdtts=float(weights[np.where(decoy_list==decoy_name)[0]])
                #reading the pdb file
                parser = PDBParser()
                with warnings.catch_warnings(record=True) as w:
                    structure = parser.get_structure("", train_data_files)
                
                # one_hot_atom is the one  hot encoding of all the atoms
                #one_hot_res is the one hot encoding of all residues
                #neigh_same_res are indices of the neighbours with respect to each source atom and have the same residue with respect to the source atom
                #neigh_diff_res are the indices of the neighbours with respect to each source atom and have  different residue with respect to the source atom
                #total_atoms is the total number of atoms present in the pdb file
                #res_feat is the total number of embeddings of each residue
                #atom_feat is the total number of embeddings of eah atom 

                one_hot_atom=atom1(structure)
                one_hot_res=res1(structure)
                neigh_same_res,neigh_diff_res=neigh1(structure)
                total_atoms=len(one_hot_atom)
                atom_feat=len(one_hot_atom[0])
                res_feat=len(one_hot_res[0])
                
                return total_atoms,atom_feat,res_feat,one_hot_atom,one_hot_res,neigh_same_res,neigh_diff_res,required_gdtts,0
            #except:
            #    print(f'Fail {self.train_data_files_list[loc1]}')
            #    return 0,0,0,0,0,0,0,0,0,0,0,0,1      

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    '''
    batch_size=len(batch)
    max_atoms=max([p2[0] for p2 in batch]) 
    one_hot_atom=torch.zeros(batch_size,max_atoms,batch[0][1])
    one_hot_res=torch.zeros(batch_size,max_atoms,batch[0][2])
    neigh_same_res=torch.zeros(batch_size,max_atoms,10).fill_(-1)
    neigh_diff_res=torch.zeros(batch_size,max_atoms,10).fill_(-1)
     
    flag=torch.zeros((batch_size))
  
    gdt_ts=torch.zeros((batch_size))
    for i in range(len(batch)):
            #print('Inside')
            #print(batch[i][8])
            if batch[i][8]==0:
                
                one_hot_atom[i][:len(batch[i][3])]=torch.tensor(batch[i][3])
                one_hot_res[i][:len(batch[i][4])]=torch.tensor(batch[i][4])
                neigh_same_res[i][:len(batch[i][5])]=torch.tensor(batch[i][5])
                neigh_diff_res[i][:len(batch[i][6])]=torch.tensor(batch[i][6])
                  
              
                gdt_ts[i]=((batch[i][7]))

    return one_hot_atom,one_hot_res,neigh_same_res,neigh_diff_res,gdt_ts
def get_dataloader(opt):

    train_dataset = protein_read(opt.train_data,opt.train_label)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batchSize,
        shuffle=False,collate_fn=collate_fn_padd,
        num_workers=int(opt.workers))

    return train_loader


class data1:
    def __init__(self,train_data_loc,train_label_loc,batch_size,workers):
        self.train_data=train_data_loc
        self.train_label=train_label_loc
        self.batchSize=batch_size
        self.workers=workers
        

if  __name__ == "__main__":

    data_loc=("data/train_data/*/*")
    label_loc=("data/train_labels/")
    temp=data1(data_loc,label_loc,30,7)

    for i1,i2,i3,i4,i5 in (get_dataloader(temp)):
       
            print(len(i1),i4[0][:15],i5)
            
