'''
Using Sklearn One hot encoder to encode the atoms
Output is of size N*M where N is the total number of atoms and M is the total number of encoded features

'''
import warnings
from Bio.PDB import *
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import glob
from biopandas.pdb import PandasPdb
def atom1(structure):
    atomslist=np.array(sorted(np.array(['C', 'CA', 'CB', 'CG', 'CH2', 'N','NH2',  'OG','OH', 'O1', 'O2', 'SE','1']))).reshape(-1,1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(atomslist)
    atom_list=[]
    for atom in structure.get_atoms():
        if atom.get_name() in atomslist:
            atom_list.append(atom.get_name())
        else:
            atom_list.append("1")
    atoms_onehot=enc.transform(np.array(atom_list).reshape(-1,1)).toarray()
    return atoms_onehot
if __name__ == "__main__":
    loc=glob.glob("data/train_data/T0759-D1/*")
    loc1=loc[0]
    print(loc1)
    parser = PDBParser()
    with warnings.catch_warnings(record=True) as w:
      structure = parser.get_structure("", loc1)
    t1=(atom1(structure))
    print(t1[0:10])
    

