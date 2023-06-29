#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from rdkit import Chem
from rdkit import DataStructs
import matplotlib.pyplot as plt
from rdkit.Chem import MACCSkeys, rdReducedGraphs, RDKFingerprint
tqdm.pandas()


# In[2]:


surechem_df = pd.read_csv('data/EDA_df.txt.gz', sep='\t', compression='gzip')


# In[3]:


surechem_df['year'] = surechem_df['PUBLICATION_DATE'].progress_apply(lambda x: x.split('-')[0])


# In[4]:


surechem_df.head(2)


# In[5]:


unique_inchi_df = surechem_df.drop_duplicates(subset=["InChIKey", "year"], keep='first')
unique_inchi_df.reset_index(drop=True, inplace=True)
len(unique_inchi_df)


# In[6]:


unique_smile_df = unique_inchi_df.drop_duplicates(subset=['SMILES'], keep='first')
unique_smile_df.head(2)


# In[7]:


len(unique_smile_df)


# In[31]:


if os.path.exists('MACCS_df.npy'):
    MACCS_df = pd.DataFrame(np.load('MACCS_df.npy', allow_pickle=True), columns=['SMILES', 'fingerprint'])
    RDK_df = pd.DataFrame(np.load('RDK_df.npy', allow_pickle=True), columns=['SMILES', 'fingerprint'])
    ErG_df = pd.DataFrame(np.load('ErG_df.npy', allow_pickle=True), columns=['SMILES', 'fingerprint'])
    processed_smiles = set(MACCS_df['SMILES'].values.tolist())
else:
    MACCS_df = pd.DataFrame(columns=['SMILES', 'fingerprint'])
    RDK_df = pd.DataFrame(columns=['SMILES', 'fingerprint'])
    ErG_df = pd.DataFrame(columns=['SMILES', 'fingerprint'])
    processed_smiles = set()

skipped_smiles = 0

MACCS_fingerprint_list = []
RDK_fingerprint_list = []
ErG_fingerprint_list = []

for index, row in tqdm(unique_smile_df.iterrows(), total=unique_smile_df.shape[0]):
    smiles = row['SMILES']

    if smiles in processed_smiles:
        skipped_smiles += 1
        continue

    try:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            skipped_smiles += 1
            continue

        MACCS_fingerprint = MACCSkeys.GenMACCSKeys(molecule)
        RDK_fingerprint = RDKFingerprint(molecule)
        ErG_fingerprint = rdReducedGraphs.GetErGFingerprint(molecule)

        MACCS_fingerprint_list.append({'SMILES': smiles, 'fingerprint': MACCS_fingerprint})
        RDK_fingerprint_list.append({'SMILES': smiles, 'fingerprint': RDK_fingerprint})
        ErG_fingerprint_list.append({'SMILES': smiles, 'fingerprint': ErG_fingerprint})

        processed_smiles.add(smiles)

    except:
        skipped_smiles += 1
        continue

    if len(MACCS_fingerprint_list) == 10 or index == unique_smile_df.shape[0] - 1:
        MACCS_data_df = pd.DataFrame(MACCS_fingerprint_list)
        RDK_data_df = pd.DataFrame(RDK_fingerprint_list)
        ErG_data_df = pd.DataFrame(ErG_fingerprint_list)

        MACCS_df = pd.concat([MACCS_df, MACCS_data_df], ignore_index=True)
        RDK_df = pd.concat([RDK_df, RDK_data_df], ignore_index=True)
        ErG_df = pd.concat([ErG_df, ErG_data_df], ignore_index=True)

        np.save('MACCS_df.npy', MACCS_df.values)
        np.save('RDK_df.npy', RDK_df.values)
        np.save('ErG_df.npy', ErG_df.values)

        MACCS_fingerprint_list = []
        RDK_fingerprint_list = []
        ErG_fingerprint_list = []

fingerprint_df = pd.DataFrame()
fingerprint_df['MACCS_fingerprint'] = MACCS_fingerprint_list
fingerprint_df['RDK_fingerprint'] = RDK_Fingerprint_list
fingerprint_df['ErG_Fingerprint'] = ErG_Fingerprint_list
fingerprint_df.head(2)for i in range(len(fingerprint_df)):

    moleculem_query = fingerprint_df['MACCS_fingerprint'][i]
    moleculem_list = fingerprint_df['MACCS_fingerprint']
    similarity_scores_maccs = DataStructs.BulkTanimotoSimilarity(moleculem_query, moleculem_list)
    fingerprint_df['tanimoto_maccs'][i] = similarity_scores_maccs

    moleculer_query = fingerprint_df['RDK_fingerprint'][i]
    moleculer_list = fingerprint_df['RDK_fingerprint']
    similarity_scores_rdKit = DataStructs.BulkTanimotoSimilarity(moleculer_query, moleculer_list)
    fingerprint_df['tanimoto_rdKit'][i] = similarity_scores_rdKit

    moleculee_query = fingerprint_df['ErG_Fingerprint'][i]
    moleculee_list = fingerprint_df['ErG_Fingerprint']
    similarity_scores_erg = DataStructs.BulkTanimotoSimilarity(moleculee_query, moleculee_list)
    fingerprint_df['tanimoto_erg'][i] = similarity_scores_ergplt.hist(fingerprint_df['tanimoto_maccs'], bins=10, alpha=0.5, label='MACCS')

plt.hist(fingerprint_df['tanimoto_rdKit'], bins=10, alpha=0.5, label='RDKit')

plt.hist(fingerprint_df['tanimoto_erg'], bins=10, alpha=0.5, label='ErG')

plt.xlabel('Tanimoto Similarity Score')
plt.ylabel('Frequency')

plt.legend()

plt.show()