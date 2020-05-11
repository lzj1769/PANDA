import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import configure

df = pd.read_csv(configure.TRAIN_DF)

print(f"total image: {len(df)}")

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
