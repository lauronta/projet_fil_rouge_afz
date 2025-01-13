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

def create_description(row, levels=[0, 1, 2, 3, 4]):
    return (' '.join([str(row.iloc[i]) for i in levels])).strip()

combinasons_pertinentes = [[0], [1], 
                            [0, 1], [0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3, 4],
                            [0, 2], [0, 3], [0, 4],
                            [0, 1, 3], [0, 1, 4],
                            [1, 2], [1, 3], [1, 4],
                            [1, 2, 3], [1, 2, 4]]
def create_database(table, combinations, nan=np.nan, replace_nan="", linker=" ", start_libelle=3, start_merge=3, end_merge=8):
    table = table.replace(nan, replace_nan)
    all_descriptions = []
    for comb in combinations:
        comb = [lib + start_libelle for lib in comb]
        all_descriptions.append(table.apply(create_description, levels=comb, axis=1))

    descriptions_db = pd.concat(all_descriptions, axis=0).to_frame()
    descriptions_db.columns = ["Descriptions"]
    
    augmented_table = pd.concat([table for i in range(len(all_descriptions))], axis=0)
    # Creation de la database sans les lbellé, avec les descriptions à la place
    if end_merge is None:
        all_db = pd.concat([augmented_table.iloc[:,:start_merge], descriptions_db],
                       axis=1)
    else:
        all_db = pd.concat([augmented_table.iloc[:,:start_merge], descriptions_db, augmented_table.iloc[:,end_merge:]],
                        axis=1)
    
    return all_db.drop_duplicates()
    
# desc_DB = create_database(table_fourrages, combinasons_pertinentes)

# desc_DB.to_excel(WD + "/TableINRA2018_AvecDescriptions.xlsx")

def write_file(table, column, path, sep='\n'):
    with open(path, 'w') as f:
        for to_write in table[column]:
            f.write(to_write + sep)

# write_file(desc_DB, "Descriptions", WD + '/Descriptions_TableINRA2018.txt')

# Goal : To Generate the database for Feedipedia

PATH_FEED = WD + "/nomenclatures_et_définitions.xlsx"
table_feedipedia = pd.read_excel(PATH_FEED, header=0)
ALL_FEED_COLS = list(table_feedipedia.columns)
INTERESTING_COLS = ["FEED_CAT_NAME", "FEED.FEED_NAME", "FEED.FEED_DEF"]
INTERESTING_COLS_IDX = [ALL_FEED_COLS.index(col) for col in INTERESTING_COLS]

desc_FEEDIPEDIA_DB = create_database(table_feedipedia, 
                                     [INTERESTING_COLS_IDX], 
                                     start_libelle=0,
                                     start_merge=5,
                                     end_merge=None)

write_file(desc_FEEDIPEDIA_DB, "Descriptions", WD + '/Descriptions_FEEDIPEDIA_FR.txt')

INTERESTING_COLS_ENG = ["FEED_CAT_NAME_ENGLISH", "FEED_WORLD.FEED_NAME", "FEED_WORLD.FEED_DEF"]
INTERESTING_COLS_ENG_IDX = [ALL_FEED_COLS.index(col) for col in INTERESTING_COLS_ENG]

desc_FEEDIPEDIA_DB_ENG = create_database(table_feedipedia, 
                                        [INTERESTING_COLS_ENG_IDX], 
                                        start_libelle=0,
                                        start_merge=1,
                                        end_merge=7)

write_file(desc_FEEDIPEDIA_DB_ENG, "Descriptions", WD + '/Descriptions_FEEDIPEDIA_ENG.txt')

