import pandas as pd
from preprocessing import transforms as T

def apply_clahe_to_row(row):
    
    clahe_transformer = T.Clahe(clipLimit=2.0)
    image = row.values.reshape(110, 110)
    clahe_image = clahe_transformer.transform(image)
    return pd.Series(clahe_image.flatten())

def apply_lr_to_row(row):
    
    lr_transformer = T.LRandomFlip()
    image = row.values.reshape(110, 110)
    lr_image = lr_transformer.transform(image)
    return pd.Series(lr_image.flatten())

def apply_ur_to_row(row):
    
    ur_transformer = T.URandomFlip()
    image = row.values.reshape(110, 110)
    ur_image = ur_transformer.transform(image)
    return pd.Series(ur_image.flatten())

def apply_d1_to_row(row):
    
    d1_transformer = T.D1RandomFlip()
    image = row.values.reshape(110, 110)
    d1_image = d1_transformer.transform(image)
    return pd.Series(d1_image.flatten())

def apply_d2_to_row(row):
    
    d2_transformer = T.D2RandomFlip()
    image = row.values.reshape(110, 110)
    d2_image = d2_transformer.transform(image)
    return pd.Series(d2_image.flatten())

def augment_df(df, labels):
    print('Starting Data Augmentation...')
    df_copy = df.copy()
    df_copy['Volcano?'] = labels
    df_volcano = df_copy[df_copy['Volcano?']==1]
    
    # CLAHE augmentation
    df_clahe = df_volcano.drop(columns='Volcano?')
    df_clahe.iloc[:, :] = df_clahe.apply(apply_clahe_to_row, axis=1)
    df_clahe['Volcano?'] = 1 
    
    # LRandom Flip augmentation
    df_lr = df_volcano.drop(columns='Volcano?')
    df_lr.iloc[:, :] = df_lr.apply(apply_lr_to_row, axis=1)
    df_lr['Volcano?'] = 1 
    
    # URandom Flip augmentation
    df_ur = df_volcano.drop(columns='Volcano?')
    df_ur.iloc[:, :] = df_ur.apply(apply_ur_to_row, axis=1)
    df_ur['Volcano?'] = 1 
    
    # D1Random Flip augmentation
    df_d1 = df_volcano.drop(columns='Volcano?')
    df_d1.iloc[:, :] = df_d1.apply(apply_d1_to_row, axis=1)
    df_d1['Volcano?'] = 1 
    
    # D2Random Flip augmentation
    df_d2 = df_volcano.drop(columns='Volcano?')
    df_d2.iloc[:, :] = df_d2.apply(apply_d2_to_row, axis=1)
    df_d2['Volcano?'] = 1 
    
    # Augment full dataset and shuffle
    aug_df = pd.concat([df_copy, df_clahe, df_lr, df_ur, df_d1, df_d2], axis=0)
    aug_df = aug_df.sample(frac=1).reset_index(drop=True)
    print(f'N. of images before augmentation: {df_copy.shape[0]}\nN. of images after augmentation: {aug_df.shape[0]}')
    print('Data Augmentation finised!')
    return aug_df.drop(columns='Volcano?'), aug_df['Volcano?']
