import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import pickle

seed = 0
for site in ['HaN', 'Lung']:
    dadta_dir = r'NII_data/train_{}'.format(site)
    ID_list = os.listdir(dadta_dir)
    isVMAT_list = []
    for ID in ID_list:
        isVMAT = pd.read_csv(os.path.join(dadta_dir, ID, 'isVMAT.csv'))['isVMAT'][0]
        isVMAT_list.append(isVMAT)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    split_data = []
    for train_index, val_index in kf.split(ID_list, isVMAT_list):
        train_ID = []
        val_ID = []
        for index in train_index:
            train_ID.append(ID_list[index])
        for index in val_index:
            val_ID.append(ID_list[index])
        split_data.append({'train': train_ID, 'val': val_ID})
    with open('NII_data/{}_split_5fold_seed{}.pkl'.format(site, seed), "wb") as f:
        pickle.dump(split_data, f)

# Merge split
HaN_pkl_data = pickle.load(open('NII_data/HaN_split_5fold_seed{}.pkl'.format(seed), 'rb'))
Lung_pkl_data = pickle.load(open('NII_data/Lung_split_5fold_seed{}.pkl'.format(seed), 'rb'))
split_data = []
for i in range(5):
    HaN_train = HaN_pkl_data[i]['train']
    Lung_train = Lung_pkl_data[i]['train']
    HaN_val = HaN_pkl_data[i]['val']
    Lung_val = Lung_pkl_data[i]['val']
    HaN_train = ['train_HaN/{}'.format(ID) for ID in HaN_train]
    HaN_val = ['train_HaN/{}'.format(ID) for ID in HaN_val]
    Lung_train = ['train_Lung/{}'.format(ID) for ID in Lung_train]
    Lung_val = ['train_Lung/{}'.format(ID) for ID in Lung_val]
    split_data.append({'train': HaN_train + Lung_train,
                       'val': HaN_val + Lung_val})

with open('all_split_5fold_seed{}.pkl'.format(seed), "wb") as f:
    pickle.dump(split_data, f)