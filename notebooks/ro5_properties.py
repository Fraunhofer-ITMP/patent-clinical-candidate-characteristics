import os
import json
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from rdkit.Chem import Descriptors, MolFromSmiles, Lipinski
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers


DATA_DIR = '../data'

def get_properties():
    """Get molecular properties from RDKit"""

    surechembl_df = pd.read_parquet(f'{DATA_DIR}/surechembl_dump.pq')
    surechembl_df['year'] = surechembl_df['PUBLICATION_DATE'].progress_apply(lambda x: x.split('-')[0])

    # Drop duplicate entries
    surechembl_df = surechembl_df.drop_duplicates(subset=["InChIKey", "year"], keep='first')

    unique_compounds = surechembl_df.SMILES.unique()

    if os.path.exists(f'{DATA_DIR}/properties.json'):
        with open(f'{DATA_DIR}/properties.json', 'r') as f:
            smiles_to_property_dict = json.load(f)
    else:
        smiles_to_property_dict = defaultdict(dict)

    c = 0

    for counter, smiles in tqdm(enumerate(unique_compounds), total=len(unique_compounds)):

        if smiles in smiles_to_property_dict:
            continue

        try:
            molecule = MolFromSmiles(smiles)
        except:
            smiles_to_property_dict[smiles] = {
                'mw': None,
                'logp': None,
                'n_hba': None,
                'n_hbd': None,
                'rot_bonds': None,
                'tpsa': None,
                'fsp3': None,
                'n_chiral': None,
            }
            continue

        try:
            molecular_weight = Descriptors.ExactMolWt(molecule)
        except:
            smiles_to_property_dict[smiles] = {
                'mw': None,
                'logp': None,
                'n_hba': None,
                'n_hbd': None,
                'rot_bonds': None,
                'tpsa': None,
                'fsp3': None,
                'n_chiral': None,
            }

            continue

        c += 1
        logp = Descriptors.MolLogP(molecule)
        n_hba = Descriptors.NumHAcceptors(molecule)
        n_hbd = Descriptors.NumHDonors(molecule)
        rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
        tpsa = Descriptors.TPSA(molecule)
        fsp3 = Lipinski.FractionCSP3(molecule)
        chiral_count = len(tuple(EnumerateStereoisomers(molecule)))

        smiles_to_property_dict[smiles] = {
            'mw': molecular_weight,
            'logp': logp,
            'n_hba': n_hba,
            'n_hbd': n_hbd,
            'rot_bonds': rotatable_bonds,
            'tpsa': tpsa,
            'fsp3': fsp3,
            'n_chiral': chiral_count,
        }

        if c == 100:
            with open(f'{DATA_DIR}/properties.json', 'w') as f:
                json.dump(smiles_to_property_dict, f, ensure_ascii=False, indent=2)
            c = 0

    with open(f'{DATA_DIR}/properties.json', 'w') as f:
        json.dump(smiles_to_property_dict, f, ensure_ascii=False, indent=2)

if __name__ == '__main__':
    get_properties()