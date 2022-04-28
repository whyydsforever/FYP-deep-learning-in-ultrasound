# %%
import os
from glob import glob
import shutil
import random

# %%
random.seed(1)
f_list = sorted(glob(os.path.join("D:\FYP\data\mask", "*.png")))
random.shuffle(f_list)
print(f_list)

# %%
fold = 5
val_length = len(f_list) // fold

# %%
for i in range(fold):
    val_files = f_list[i*val_length : (i+1)*val_length]
    train_files = f_list[:i*val_length] + f_list[(i+1)*val_length:]
    print(i, val_files, train_files)

    fold_dir = f"./data/mask_{i}"
    if not os.path.exists(fold_dir):
     os.mkdir(fold_dir)
     os.mkdir(os.path.join(fold_dir, "val"))
     os.mkdir(os.path.join(fold_dir, "train"))

    for f in val_files:
     shutil.copy(f, os.path.join(fold_dir, "val/"))
    for f in train_files:
     shutil.copy(f, os.path.join(fold_dir, "train/"))