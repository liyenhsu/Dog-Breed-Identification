import os
import shutil
import pandas as pd

df = pd.read_csv('labels.csv')
path = 'train_data'

if os.path.exists(path):
    shutil.rmtree(path)

for _, (filename, breed) in df.iterrows():
    folder = path + '/' + breed
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy('train/' + filename + '.jpg', folder)
    
df = pd.read_csv('sample_submission.csv')
path = 'test_data'

if os.path.exists(path):
    shutil.rmtree(path)

for filename in df.id:
    folder = path + '/' + 'unknown'
    if not os.path.exists(folder):
        os.makedirs(folder)
    shutil.copy('test/' + filename + '.jpg', folder)

