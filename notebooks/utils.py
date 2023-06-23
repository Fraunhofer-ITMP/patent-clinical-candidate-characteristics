# coding: utf-8

import os
import pandas as pd
from tqdm import tqdm

def find_repurposed_compounds(data_df: pd.DataFrame) -> pd.DataFrame:
    """Find compounds that have been patented in previous years."""

    if os.path.exists('../data/repeated_compound.tsv'):
        return pd.read_csv('../data/repeated_compound.tsv', sep='\t')

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
    c.to_csv('../data/repeated_compound.tsv', sep='\t', index=False)
    return c