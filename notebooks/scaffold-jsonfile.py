#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from tqdm import tqdm
import rdkit
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import MolFromSmiles
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
tqdm.pandas()
from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*')


surechem_df = pd.read_csv('data/EDA_df.txt.gz', sep='\t', compression='gzip')

surechem_df['year'] = surechem_df['PUBLICATION_DATE'].progress_apply(lambda x: x.split('-')[0])


surechem_df.head(2)

unique_inchi_df = surechem_df.drop_duplicates(subset=["InChIKey","year"], keep='first')
unique_inchi_df.reset_index(drop=True, inplace=True)
len(unique_inchi_df)


smiles_df = unique_inchi_df[['SMILES', 'year', 'PATENT_ID' ]]


scaffold_list = []
skipped_smiles = 0

for smiles, year, patent_id in tqdm(smiles_df.values):
    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            skipped_smiles += 1
            continue
        
        scaffold_smiles = Chem.MolToSmiles(GetScaffoldForMol(molecule)) 
        
        if scaffold_smiles == '':
            generic_scaffold_smiles = Chem.MolToSmiles(rdkit.Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric(Chem.MolFromSmiles(smiles)))
            scaffold_list.append({'scaffold': generic_scaffold_smiles, 'year': year, 'patent_ID': patent_id })
            
            continue
        
        scaffold_list.append({'scaffold': scaffold_smiles, 'year': year, 'patent_ID': patent_id})
    except:
        skipped_smiles += 1
        continue

skipped_smiles, len(scaffold_list)
# (44893, 21812332)

with open('data/scaffold_smiles.json', 'w') as f:
    json.dump(scaffold_list, f, ensure_ascii=False, indent=2)