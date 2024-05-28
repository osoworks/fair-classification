import os
import sys
import pandas as pd
sys.path.insert(0, '../../fair_classification/') # the code for fair classification is in this directory
import urllib.request
import utils as ut
import numpy as np
from random import seed, shuffle
import csv

SEED = 1122334455
seed(SEED)
np.random.seed(SEED)

train_frac = 0.7
random_state=42 

def _get_train_test_split(n_examples, train_fraction, seed, power_of_two=False):
    """
    Args:
        n_examples: Number of training examples
        train_fraction: Fraction of data to use for training
        seed: Seed for random number generation (reproducability)
        power_of_two: Whether to select the greatest power of two for training
            set size and use the remaining examples for testing.

    Returns:
        training indices, test indices
    """
    np.random.seed(seed)
    idx = np.random.permutation(n_examples)
    pivot = int(n_examples * train_fraction)
    if power_of_two:
        pivot = 2**(len(bin(pivot)) - 3)
    training_idx = idx[:pivot]
    test_idx = idx[pivot:]
    return training_idx, test_idx
    
    
def _apply_train_test_split(x, y, z, training_idx, test_idx):
    """
    Apply the train test split to the data.

    Args:
        x: Features
        y: Labels
        z: Sensitive attributes
        training_idx: Training set indices
        test_idx: Test set indices

    Returns:
        Xtr, Xte, ytr, yte, Ztr, Zte: X are the features, y are the labels and Z
            are the sensitive attributes. 'tr' stands for training and 'te' for
            test set.
    """
    Xtr = x[training_idx, :]
    Xte = x[test_idx, :]
    ytr = y[training_idx]
    yte = y[test_idx]
    Ztr = z[training_idx, :]
    Zte = z[test_idx, :]
    return Xtr, Xte, ytr, yte, Ztr, Zte




def check_data_file(fname):
    files = os.listdir(".")
    print("Looking for file '%s' in the current directory..." % fname)

    if fname not in files:
        print("'%s' not found! Downloading from UCI Archive..." % fname)
        addr = "https://archive.ics.uci.edu/static/public/222/data.csv"
        response = urllib.request.urlopen(addr)
        data = response.read().decode('utf-8')
        with open(fname, "w") as fileOut:
            fileOut.write(data)
        print("'%s' downloaded and saved locally.." % fname)
    else:
        print("File found in current directory..")

    print()
    return

def load_bank_data(load_data_size=None):

    data_files = ['data.csv']  

    for f in data_files:
      check_data_file(f)

      with open(f, newline='') as csvfile:
          reader = csv.reader(csvfile)

    bank = pd.read_csv("data.csv")
    
    bank['marital'].loc[bank['marital']!='married']=0
    bank['marital'].loc[bank['marital']=='married']=1
    attrs = bank.columns
#     attrs = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#        'previous', 'poutcome', 'y'] # all attributes
    int_attrs = ['age', 'balance','day','duration','campaign','pdays','previous'] # attributes with integer values -- the rest are categorical
    sensitive_attrs = ['marital'] # the fairness constraints will be used for this feature
    attrs_to_ignore = ['marital','day', 'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y'] # sex and race are sensitive feature so we will not use them in classification, we will not consider fnlwght for classification since its computed externally and it highly predictive for the class (for details, see documentation of the adult data)
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)


    X = []
    y = []
    x_control = {}

    attrs_to_vals = {} # will store the values for each attribute for all users
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []


    bank['y'] = bank['y'].map({"no": 0, "yes": 1})
    y = bank.values[:,-1]


    for i in range(len(bank)):
        line = bank.iloc[i].values
#         class_label = line[-1]
#         if class_label in ["no"]:
#             class_label = -1
#         elif class_label in ["yes"]:
#             class_label = +1
#         else:
#             raise Exception("Invalid class label value")

#         y.append(class_label)
#         if i ==0:
#             print(line)
#             print(len(line))
        for j in range(0,len(line)):
            attr_name = attrs[j]
            attr_val = line[j]
                # reducing dimensionality of some very sparse features
#             if attr_name == 'previous':
#                 print(attr_name,": ",line)
#             if i ==0:
#                 print(attr_name, attr_val)
            if attr_name in sensitive_attrs:
                x_control[attr_name].append(attr_val)
            elif attr_name in attrs_to_ignore:
                pass
            else:
                attrs_to_vals[attr_name].append(attr_val)
#     print(bank['previous'])
#     print(attrs_to_vals['previous'])


    def convert_attrs_to_ints(d):
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue
            #모든 값을 strings로 변환
            attr_vals = [str(val) for val in attr_vals]
            uniq_vals = sorted(list(set(attr_vals)))

            val_dict = {}
            for i, val in enumerate(uniq_vals):
                val_dict[val] = i

            # replace the values with their integer encoding
            for i in range(0,len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    
    # convert the discrete values to their integer representations
    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)


    # if the integer vals are not binary, we need to get one-hot encoding for them
    for attr_name in attrs_for_classification:
        
        attr_vals = attrs_to_vals[attr_name]
#         print(attr_name)
#         print(attr_vals[0])
        if attr_name in int_attrs or attr_name == "native_country": # the way we encoded native country, its binary now so no need to apply one hot encoding on it
#             if attr_vals == []:
#                 print(attr_name)
            X.append(attr_vals)
#             print(attr_name,attr_vals)

        else:
            
            attr_vals, index_dict = ut.get_one_hot_encoding(attr_vals)
#             print(attr_vals[0])
#             print(attr_vals.shape)
            if attr_vals.shape==(45211,):
                attr_vals=attr_vals.reshape(45211,1)
            for inner_col in attr_vals.T:                
                X.append(inner_col)
#                 if inner_col == []:
#                     print(attr_name)
                
    for i,xx in enumerate(X):
        if np.array(xx).shape != (45211,):
            print(i,np.array(xx).shape)
#     print(len(X),len(X[0]))
#     print(X[:3])
    # convert to numpy arrays for easy handline
    X = np.array(X, dtype=float).T
    y = np.array(y, dtype = float)
    for k, v in x_control.items(): x_control[k] = np.array(v, dtype=float)
        
    # shuffle the data
    perm = list(range(0,len(y))) # shuffle the data before creating each fold
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    # see if we need to subsample the data
    if load_data_size is not None:
        print ("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]
#     print(x_control)
    n = X.shape[0]
    # Create train test split
    Z = np.expand_dims(x_control[k],axis=-1)
    tr_idx, te_idx = _get_train_test_split(n, train_frac, random_state)
    Xtr, Xte, ytr, yte, Ztr, Zte = _apply_train_test_split(X, y, Z,
                                                           tr_idx, te_idx)


    return Xtr, Xte, ytr, yte, Ztr, Zte




# 데이터를 로드하고 결과를 출력하는 코드 추가
if __name__ == "__main__":
    Xtr, Xte, ytr, yte, Ztr, Zte = load_bank_data()  # 데이터 로드 (필요에 따라 사이즈 조정 가능)
    print("Xtr shape:", Xtr.shape)
    print("ytr shape:", ytr.shape)
    print("Xte shape:", Xte.shape)
    print("yte shape:", yte.shape)
    print("Sensitive attribute marital distribution in training set:", np.bincount(Ztr[:, 0].astype(int)))
    print("Sensitive attribute marital distribution in test set:", np.bincount(Zte[:, 0].astype(int)))
