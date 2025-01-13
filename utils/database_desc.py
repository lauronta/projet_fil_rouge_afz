import os 
import numpy as np 
import pandas as pd
from itertools import combinations

# Goal : To generate the description database for INRA2018 table
WD = os.getcwd()
PATH_DB = WD + "/INRA2018_TablesFourrages_etude_prediction_20241121.xlsx"
table_fourrages = pd.read_excel(PATH_DB, header=1)
niveaux_libelles = [col for col in table_fourrages.columns if 'Libellé' in col]
table_libelles = table_fourrages[niveaux_libelles]
print(table_libelles.head())

def create_description(row, levels=[0, 1, 2, 3, 4]):
    return (' '.join([str(row.iloc[i]) for i in levels])).strip()

combinasons_pertinentes = [[0], [1], 
                            [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4],
                            [0, 2], [0, 3], [0, 4],
                            [0, 1, 3], [0, 1, 4],
                            [1, 2], [1, 3], [1, 4],
                            [1, 2, 3], [1, 2, 4]]
def create_database(table, combinations, nan=np.nan, replace_nan="", linker=" ", start_libelle=3):
    table = table.replace(nan, replace_nan)
    all_descriptions = []
    for comb in combinations:
        comb = [lib + start_libelle for lib in comb]
        all_descriptions.append(table.apply(create_description, levels=comb, axis=1))

    descriptions_db = pd.concat(all_descriptions, axis=0).to_frame()
    descriptions_db.columns = ["Descriptions"]
    
    augmented_table = pd.concat([table for i in range(len(all_descriptions))], axis=0)
    # Creation de la database sans les lbellé, avec les descriptions à la place
    all_db = pd.concat([augmented_table.iloc[:,:3], descriptions_db, augmented_table.iloc[:,8:]],
                       axis=1)
    
    return all_db.drop_duplicates()
    
desc_DB = create_database(table_fourrages, combinasons_pertinentes)

# desc_DB.to_excel(WD + "/TableINRA2018_AvecDescriptions.xlsx")

with open(WD + '/Descriptions_TableINRA2018.txt', 'w') as f:
    for desc in desc_DB["Descriptions"]:
        f.write(desc + '\n')

# Goal : To Generate the database for Feedipedia

