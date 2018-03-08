import pandas as pd
import numpy as np



#####
##SET VARIABLES
#####
in_file='/Volumes/HOLOCRON/corpora/mohler/all_data11-9.csv'
out_file='/Users/ZaqRosen/Documents/Corpora/mohler/all_data11-9.csv'



#####
##IMPORT DATA
#####
cols=['tref', 'subj', 'dobj', 'syn', 'verb', 'obl1', 'obl2', 'Labels']
df_in = pd.read_csv(in_file, names=cols, skipinitialspace=True)



#####
##FUNCITONS
#####
def get_good_idx(bad_idx, dfk=df_in):
        wanted_idx=[]
        for k in range(len(dfk)):
                if k not in bad_idx:
                        wanted_idx.append(k)
        return wanted_idx



#####
##IMPLEMENTATION
#####
nans=df_in.index[df_in['Labels'].isin([np.nan])].values.tolist()
good_idx=get_good_idx(nans)
df_sans=df_in.loc[good_idx]
df_sans.to_csv(out_file, sep=',', header=False, index=False, encoding='utf-8')
