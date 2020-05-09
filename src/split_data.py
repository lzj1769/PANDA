import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import configure

low_quality_images = ['033e39459301e97e457232780a314ab7',
                      '0b6e34bf65ee0810c1a4bf702b667c88',
                      '3385a0f7f4f3e7e7b380325582b115c9',
                      '3790f55cad63053e956fb73027179707',
                      '5204134e82ce75b1109cc1913d81abc6',
                      'a08e24cff451d628df797efc4343e13c']

df = pd.read_csv(configure.TRAIN_DF)

print(f"total image: {len(df)}")

# drop low quality images
for image_id in low_quality_images:
    df = df[df['image_id'] != image_id]

print(f"removed low quality images: {len(df)}")

# fix mislabelled information
# df.replace(df.loc[7273]['isup_grade'], 3, inplace=True)

# change negative to '0+0'
df['gleason_score'] = df['gleason_score'].apply(lambda x: "0+0" if x == "negative" else x)
data_provider = {'karolinska': 0, 'radboud': 1}

df = df.replace({'data_provider': data_provider})

X = df.index
y = df['isup_grade'].values.tolist()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=2020)
for i, (train_index, valid_index) in enumerate(kf.split(X, y)):
    df_train, df_valid = df.iloc[train_index], df.iloc[valid_index]
    df_train.to_csv(os.path.join(configure.SPLIT_FOLDER, "fold_{}_train.csv".format(i)))
    df_valid.to_csv(os.path.join(configure.SPLIT_FOLDER, "fold_{}_valid.csv".format(i)))
