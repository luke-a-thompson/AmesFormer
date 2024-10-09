import pandas as pd
from math import floor

honma = pd.read_excel("/home/luke/ames_graphormer/data/raw/Honma_New.xlsx")
hansen = pd.read_excel("datasets/Hansen_New.xlsx")[:-1589]
isssty = pd.read_excel("datasets/ISSSTY_fixed.xlsx")
eurl = pd.read_excel("datasets/eurl_ecvam.xls", skiprows=1)

combined = pd.read_csv('combined_datasets/Combined_2s_as_0s.csv')[:-1589]


print(f"Honma original length: {len(honma)}")
print(f"EURL original length: {len(eurl)}")
print(f"ISSSTY original length: {len(isssty)}")
print(f"Hansen original length: {len(hansen)}")
print(f"Combined original length: {len(combined)}")

print(f"Combined train: {floor(len(combined[combined['split'] == 'Train/Validation']) * 0.8)}")
print(f"Combined validation: {floor(len(combined[combined['split'] == 'Train/Validation']) * 0.2)}")
print(f"Combined test: {len(combined[combined['split'] == 'Test'])}")
print(combined['source'].value_counts())

print(f"Combined train: {floor(len(combined[combined['split'] == 'Train/Validation']) * 0.8)}")
print(f"Combined validation: {floor(len(combined[combined['split'] == 'Train/Validation']) * 0.2)}")
print(f"Combined test: {len(combined[combined['split'] == 'Test'])}")

# Calculate the counts for ames==1 and ames==0 in train and validation splits
train_val = combined[combined['split'] == 'Train/Validation']
train_size = floor(len(train_val) * 0.8)
validation_size = len(train_val) - train_size

# Randomly shuffle the train_val dataframe
train_val_shuffled = train_val.sample(frac=1, random_state=42).reset_index(drop=True)

train = train_val_shuffled.iloc[:train_size]
validation = train_val_shuffled.iloc[train_size:]

print("\nPercentages:")
print("Train:")
print(train['ames'].value_counts(normalize=True) * 100)
print("\nValidation:")
print(validation['ames'].value_counts(normalize=True) * 100)
