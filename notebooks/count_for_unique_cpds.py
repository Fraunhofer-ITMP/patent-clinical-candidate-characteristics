#!/usr/bin/env python
# coding: utf-8

import pandas as pd

col = ['SureChEMBL_ID', 'SMILES', 'PUBLICATION_DATE']

file = pd.read_csv('data/EDA_df.txt.gz', sep='\t', compression='gzip', usecols = col)


year_list = []
for row in file['PUBLICATION_DATE']:
    g = row.split('-')[0]
    year_list.append(g) 


k = pd.DataFrame(year_list)
k.columns = ['Year']


file['Year'] = k['Year']
file.head(2)


unique_smile_id = file.drop_duplicates(subset=["SMILES", "Year"], keep='first')
unique_smile_id.reset_index(drop=True, inplace=True)


def define_year(year):
    df_year = unique_smile_id[(unique_smile_id['Year'] == str(year))]
    return df_year                           

def converting_to_list(df_year):
    cpds_list_year = df_year.SMILES.values.tolist()
    count_for_year = len(cpds_list_year)
    return cpds_list_year


def intersect(list1,list2):
    s1=set(list1)
    s2=set(list2)
    intersection= s2.intersection(s1)
    return intersection 

count = []
concat_df = None

for year in range(2015, 2023):
    current_df = define_year(str(year))
    cpds_list_year = converting_to_list(current_df)
    
    if year == 2015:
        count.append({'year': year, 'count': len(cpds_list_year)})
        concat_df = current_df
    else:
        cpds_previous_year = converting_to_list(concat_df)
        
        intersection = intersect(cpds_previous_year, cpds_list_year)
        count.append({'year': year, 'count': len(intersection)})
        
        concat_df = pd.concat([concat_df, current_df])
        concat_df.drop_duplicates(keep='first', inplace=True)
        concat_df.reset_index(drop=True, inplace=True)


c = pd.DataFrame(count)
c.to_csv('repeated_smiles.tsv', index=False)