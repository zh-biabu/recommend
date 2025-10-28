import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil

def load_inter_file(inter_path):
    # 假设格式为userID,itemID,rating[,timestamp]
    df = pd.read_csv(inter_path, sep="\t")
    return df

def create_id_mapping(df, user_col='userID', item_col='itemID', out_dir=None):
    user_ids = sorted(df[user_col].unique())
    item_ids = sorted(df[item_col].unique())
    user2id = {u: i for i, u in enumerate(user_ids)}
    item2id = {i: j for j, i in enumerate(item_ids)}
    df[user_col] = df[user_col].map(user2id)
    df[item_col] = df[item_col].map(item2id)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame({user_col: user_ids}).to_csv(os.path.join(out_dir, f'u_id_mapping.csv'), index=False)
        pd.DataFrame({item_col: item_ids}).to_csv(os.path.join(out_dir, f'i_id_mapping.csv'), index=False)
    return df, user2id, item2id

# def split_data(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, time_col=None, user_col="userID"):
#     # 按user分组切分
#     train_list, val_list, test_list = [], [], []
#     for user, user_df in df.groupby(user_col):
#         if time_col and time_col in user_df.columns:
#             user_df = user_df.sort_values(time_col)
#         n = len(user_df)
#         n_train = int(n * train_ratio)
#         n_val = int(n * val_ratio)
#         n_test = n - n_train - n_val
#         if n_val < 1: n_val = 1
#         if n_test < 1: n_test = 1
#         n_train = n - n_val - n_test
#         user_df = user_df.reset_index(drop=True)
#         train_list.append(user_df.iloc[:n_train])
#         val_list.append(user_df.iloc[n_train:n_train+n_val])
#         test_list.append(user_df.iloc[n_train+n_val:])
#     train = pd.concat(train_list, ignore_index=True)
#     val = pd.concat(val_list, ignore_index=True)
#     test = pd.concat(test_list, ignore_index=True)
#     return train, val, test

def split_data(df):
    train = df.loc[df["x_label"] == 0,: ]
    val = df.loc[df["x_label"] == 1,: ]
    test = df.loc[df["x_label"] == 2,: ]
    return train, val, test


def save_split(train, val, test, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    train.to_csv(os.path.join(out_dir, 'train.csv'), index=False)
    val.to_csv(os.path.join(out_dir, 'val.csv'), index=False)
    test.to_csv(os.path.join(out_dir, 'test.csv'), index=False)

if __name__ == '__main__':
    # 示例：处理baby数据集
    base_dir = os.path.join(os.path.dirname(__file__), "..", "..", "autodl-tmp", "data", 'ori_data', 'baby')
    print(os.path.abspath(base_dir))
    inter_path = os.path.join(base_dir, 'baby.inter')
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "autodl-tmp", "data", "scale_data", "baby")
    df = load_inter_file(inter_path)
    print(df.columns)
    df, user2id, item2id = create_id_mapping(df, user_col="userID", item_col="itemID", out_dir=out_dir)
    train, val, test = split_data(df)
    save_split(train, val, test, out_dir)
    # for file in os.listdir(base_dir):
    #     if file.endswith('.npy'):
    #         file_name = os.path.basename(file)
    #         target_file = os.path.join(os.path.join(os.path.dirname(__file__), "scale_data", "baby"),  file_name)
    #         source_file = os.path.join(os.path.join(os.path.dirname(__file__), "ori_data", "baby"),  file_name)
    #         shutil.copy2(source_file, target_file)
