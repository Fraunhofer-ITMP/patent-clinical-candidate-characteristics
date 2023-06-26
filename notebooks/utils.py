# coding: utf-8

import os
import sqlite3
import pandas as pd
from tqdm import tqdm

DATA_DIR = '../data'

def find_repurposed_compounds(data_df: pd.DataFrame) -> pd.DataFrame:
    """Find compounds that have been patented in previous years."""

    if os.path.exists(f'{DATA_DIR}/repeated_compound.tsv'):
        return pd.read_csv(f'{DATA_DIR}repeated_compound.tsv', sep='\t')

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
    c.to_csv(f'{DATA_DIR}/repeated_compound.tsv', sep='\t', index=False)
    return c


def cross_references():
    """Get external references (pubchem and chembl ids) for surechembl compounds."""

    surechem_df = pd.read_parquet(f'{DATA_DIR}/surechembl_dump.pq')
    inchikeys_of_interest = surechem_df['InChIKey'].unique().tolist()

    pubchem_df = pd.read_csv(
        f'{DATA_DIR}/CID-InChI-Key.gz', sep='\t', compression='gzip', names=['cid', 'inchi', 'inchikey']
    )
    pubchem_df = pubchem_df[pubchem_df['inchikey'].isin(inchikeys_of_interest)]  # 10129899 compounds
    pubchem_df.to_parquet(f'{DATA_DIR}/surechembl_pubchem_map.pq.gzip', compression='gzip')

    """ChEMBL ID to InChIKey mapping"""
    conn = sqlite3.connect(f'{DATA_DIR}/chembl_32.db')

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
    chembl_df = chembl_df[chembl_df['inchikey'].isin(inchikeys_of_interest)]
    chembl_df.to_parquet(f'{DATA_DIR}/chembl.pq.gzip', compression='gzip')



