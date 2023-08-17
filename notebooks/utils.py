# coding: utf-8

import os
import logging
import multiprocessing as mp
import json
import sqlite3
from collections import defaultdict
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import MACCSkeys, rdReducedGraphs, RDKFingerprint, MolFromSmiles, Descriptors, Lipinski
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers

from matplotlib import pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from itertools import chain

RAW_DIR = '../data/raw'
MAPPING_DIR = '../data/mappings'
PROCESSED_DIR = '../data/processed'
FIG_DIR = '../data/figures'

os.makedirs(PROCESSED_DIR, exist_ok=True)

logger = logging.getLogger(__name__)

tqdm.pandas()


def find_repurposed_compounds(data_df: pd.DataFrame) -> pd.DataFrame:
    """Find compounds that have been patented in previous years."""

    if os.path.exists(f'{PROCESSED_DIR}/repeated_compound.tsv'):
        return pd.read_csv(f'{PROCESSED_DIR}/repeated_compound.tsv', sep='\t')

    df = data_df.drop_duplicates(subset=["InChIKey", "year"], keep='first')
    year_range = df['year'].unique().tolist()
    year_range.sort()

    # 2015 as the base year
    base_compounds = set(df[df['year'] == '2015']['InChIKey'].unique().tolist())

    count = [{'year': '2015', 'repuroposed_drug_count': 0, 'unique_compounds': len(base_compounds)}]

    for year1, year2 in tqdm(zip(year_range, year_range[1:]), total=len(year_range) - 1):

        previous_year_compound = set(
            df[(df['year'] == str(year1))]['InChIKey'].unique().tolist()
        )
        current_year_compound = set(
            df[(df['year'] == str(year2))]['InChIKey'].unique().tolist()
        )

        base_compounds = base_compounds.union(previous_year_compound)

        unique_cmps = current_year_compound.difference(base_compounds)
        common_cmps = current_year_compound.intersection(base_compounds)
        
        count.append({
            'year': year2, 
            'repuroposed_drug_count': len(common_cmps),
            'unique_compounds': len(unique_cmps)
        })
        
    c = pd.DataFrame(count)
    c.to_csv(f'{PROCESSED_DIR}/repeated_compound.tsv', sep='\t', index=False)
    return c


def cross_references(from_pubchem: bool = False, from_chembl: bool = False):
    """Get external references (pubchem and chembl ids) for surechembl compounds."""

    surechem_df = pd.read_parquet(f'{RAW_DIR}/surechembl_dump.pq')
    inchikeys_of_interest = surechem_df['InChIKey'].unique().tolist()

    if from_pubchem:
        pubchem_df = pd.read_csv(
            f'{MAPPING_DIR}/CID-InChI-Key.gz', sep='\t', compression='gzip', names=['cid', 'inchi', 'inchikey']
        )
        pubchem_df = pubchem_df[pubchem_df['inchikey'].isin(inchikeys_of_interest)]  # 10129899 compounds
        pubchem_df.to_parquet(f'{PROCESSED_DIR}/surechembl_pubchem_map.pq.gzip', compression='gzip')

    """ChEMBL ID to InChIKey mapping"""
    if from_chembl:
        conn = sqlite3.connect(f'{RAW_DIR}/chembl_32.db')

        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        try:
            assert len(cursor.fetchall()) > 1
        except AssertionError:
            print('Incorrect database. Please download the database again.')

        _sql = """
        SELECT
            MOLECULE_DICTIONARY.chembl_id as chembl_id,
            MOLECULE_DICTIONARY.max_phase as clinical_phase,
            COMPOUND_STRUCTURES.STANDARD_INCHI_KEY as inchikey
        FROM MOLECULE_DICTIONARY
        JOIN COMPOUND_STRUCTURES ON MOLECULE_DICTIONARY.molregno = COMPOUND_STRUCTURES.molregno
        """

        chembl_df = pd.read_sql(_sql, con=conn)
        chembl_df.to_parquet(f'{PROCESSED_DIR}/chembl.pq.gzip', compression='gzip')

"""Parallelization of compound fingerprint calucation"""

def _calculate_fingerprints(smiles: str):
    """Get the different fingerprints for compounds."""

    try:
        molecule = Chem.MolFromSmiles(smiles)
    except:
        return smiles, {'maccs': None, 'rdkit': None, 'erg': None}
    
    if molecule is None:
        return smiles, {'maccs': None, 'rdkit': None, 'erg': None}

    maccs = MACCSkeys.GenMACCSKeys(molecule).ToBitString()
    rdkit = RDKFingerprint(molecule).ToBitString()
    erg = rdReducedGraphs.GetErGFingerprint(molecule).ToBitString()

    return smiles, {'maccs': maccs, 'rdkit': rdkit, 'erg': erg}


def get_fingerprints():
    pool = mp.Pool(10)  # We have 24 cores on our linux machine

    surechembl_df = pd.read_parquet(f'{RAW_DIR}/surechembl_dump.pq')
    logger.warning(f'Loading data')
    surechembl_df['year'] = surechembl_df['PUBLICATION_DATE'].progress_apply(lambda x: x.split('-')[0])
    logger.warning(f'Loading completed')

    # Drop duplicate entries
    surechembl_df = surechembl_df.drop_duplicates(subset=["InChIKey", "year"], keep='first')

    global smiles_to_fingerprint_dict

    smiles_to_fingerprint_dict = defaultdict(dict)

    unique_compounds = surechembl_df.SMILES.unique().tolist()
    
    results = []

    # Parallelize the process
    for row in tqdm(unique_compounds, total=len(unique_compounds), desc='Extracting molecular properties'):
        results.append(pool.apply_async(_calculate_fingerprints, args=(row,)))
    
    pool.close()
    pool.join()
    
    for res in results:
        result = res.get()
        smiles_to_fingerprint_dict[result[0]] = result[1]
    
    
    with open(f'{MAPPING_DIR}/fingerprint.json', 'w') as f:
        json.dump(smiles_to_fingerprint_dict, f, ensure_ascii=False, indent=2)


"""Parallelization of compound properties calucation"""

def _calcualte_props(smiles: str):
    """Get molecular properties from RDKit"""
    
    try:
        molecule = MolFromSmiles(smiles)
    except:
        return smiles, {
            'mw': None,
            'logp': None,
            'n_hba': None,
            'n_hbd': None,
            'rot_bonds': None,
            'tpsa': None,
            'fsp3': None,
            'n_chiral': None,
        }

    try:
        molecular_weight = Descriptors.ExactMolWt(molecule)
    except:
        return smiles, {
            'mw': None,
            'logp': None,
            'n_hba': None,
            'n_hbd': None,
            'rot_bonds': None,
            'tpsa': None,
            'fsp3': None,
            'n_chiral': None,
        }

    logp = Descriptors.MolLogP(molecule)
    n_hba = Descriptors.NumHAcceptors(molecule)
    n_hbd = Descriptors.NumHDonors(molecule)
    rotatable_bonds = Descriptors.NumRotatableBonds(molecule)
    tpsa = Descriptors.TPSA(molecule)
    fsp3 = Lipinski.FractionCSP3(molecule)
    chiral_count = len(tuple(EnumerateStereoisomers(molecule)))

    return smiles, {
        'mw': molecular_weight,
        'logp': logp,
        'n_hba': n_hba,
        'n_hbd': n_hbd,
        'rot_bonds': rotatable_bonds,
        'tpsa': tpsa,
        'fsp3': fsp3,
        'n_chiral': chiral_count,
    }


def get_properties():
    pool = mp.Pool(10)  # We have 24 cores on our linux machine

    surechembl_df = pd.read_parquet(f'{RAW_DIR}/surechembl_dump.pq')
    logger.warning(f'Loading data')
    surechembl_df['year'] = surechembl_df['PUBLICATION_DATE'].progress_apply(lambda x: x.split('-')[0])
    logger.warning(f'Loading completed')

    # Drop duplicate entries
    surechembl_df = surechembl_df.drop_duplicates(subset=["InChIKey", "year"], keep='first')

    global smiles_to_property_dict

    smiles_to_property_dict = defaultdict(dict)

    unique_compounds = surechembl_df.SMILES.unique().tolist()
    
    results = []

    # Parallelize the process
    for row in tqdm(unique_compounds, total=len(unique_compounds), desc='Extracting molecular properties'):
        results.append(pool.apply_async(_calcualte_props, args=(row,)))
    
    pool.close()
    pool.join()
    
    for res in results:
        result = res.get()
        smiles_to_property_dict[result[0]] = result[1]
    
    
    with open(f'{MAPPING_DIR}/properties.json', 'w') as f:
        json.dump(smiles_to_property_dict, f, ensure_ascii=False, indent=2)


"""Venn diagram """


def get_labels(data, fill="number"):
    
    """method to get labels for venn diagram"""
    
    N = len(data)

    sets_data = [set(data[i]) for i in range(N)]  
    s_all = set(chain(*data))                             

    set_collections = {}
    for n in range(1, 2**N):
        key = bin(n).split('0b')[-1].zfill(N)
        value = s_all
        sets_for_intersection = [sets_data[i] for i in range(N) if  key[i] == '1']
        sets_for_difference = [sets_data[i] for i in range(N) if  key[i] == '0']
        for s in sets_for_intersection:
            value = value & s
        for s in sets_for_difference:
            value = value - s
        set_collections[key] = value

    if fill == "number":
        labels = {k: len(set_collections[k]) for k in set_collections}
    elif fill == "logic":
        labels = {k: k for k in set_collections}
    elif fill == "both":
        labels = {k: ("%s: %d" % (k, len(set_collections[k]))) for k in set_collections}
    else:  
        raise Exception("invalid value for fill")

    return labels


def venn4(data=None, names=None, fill="number", show_names=True, figsize: tuple=(10, 10)):
    
    """Formatting venn diagram text, name and orientation"""

    alignment = {'horizontalalignment':'center', 'verticalalignment':'baseline'}
    colors = ['r', 'b', 'g', 'c']

    if (data is None) or len(data) != 4:
        raise Exception("length of data should be 4!")
    if (names is None) or (len(names) != 4):
        names = ("set 1", "set 2", "set 3", "set 4")

    labels = get_labels(data, fill=fill)

    fig = plt.figure(figsize=figsize)  
    ax = fig.gca()
    patches = []
    width, height = 170, 110  
    patches.append(Ellipse((170, 170), width, height, -45, color=colors[0], alpha=0.5))
    patches.append(Ellipse((200, 200), width, height, -45, color=colors[1], alpha=0.5))
    patches.append(Ellipse((200, 200), width, height, -135, color=colors[2], alpha=0.5))
    patches.append(Ellipse((230, 170), width, height, -135, color=colors[3], alpha=0.5))
    for e in patches:
        ax.add_patch(e)
    ax.set_xlim(80, 320); ax.set_ylim(80, 320)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in fig.gca().spines.values():
        spine.set_visible(False)

    # 1
    pylab.text(120, 200, labels['1000'], **alignment, fontsize=18)
    pylab.text(280, 200, labels['0100'], **alignment, fontsize=18)
    pylab.text(155, 250, labels['0010'], **alignment, fontsize=18)
    pylab.text(245, 250, labels['0001'], **alignment, fontsize=18)

    # 2
    pylab.text(200, 115, labels['1100'], **alignment, fontsize=18)
    pylab.text(140, 225, labels['1010'], **alignment, fontsize=18)
    pylab.text(145, 155, labels['1001'], **alignment, fontsize=18)
    pylab.text(255, 155, labels['0110'], **alignment, fontsize=18)
    pylab.text(260, 225, labels['0101'], **alignment, fontsize=18)
    pylab.text(200, 240, labels['0011'], **alignment, fontsize=18)

    # 3
    pylab.text(235, 205, labels['0111'], **alignment, fontsize=18)
    pylab.text(165, 205, labels['1011'], **alignment, fontsize=18)
    pylab.text(225, 135, labels['1101'], **alignment, fontsize=18)
    pylab.text(175, 135, labels['1110'], **alignment, fontsize=18)

    # 4
    pylab.text(200, 175, labels['1111'], **alignment, fontsize=18)

    if show_names:
        pylab.text(110, 110, names[0], fontsize=18, **alignment)
        pylab.text(290, 110, names[1], fontsize=18, **alignment)
        pylab.text(270, 275, names[2], fontsize=18, **alignment)
        pylab.text(130, 275, names[3], fontsize=18, **alignment)

    pylab.tight_layout()
    pylab.savefig(f'{FIG_DIR}/figure_2.png', dpi=400)
    pylab.show()
