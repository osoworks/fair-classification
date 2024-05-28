import os
import sys
import urllib.request
import numpy as np
from random import seed, shuffle

SEED = 1122334455
seed(SEED)
np.random.seed(SEED)

def get_one_hot_encoding(in_arr):
    """
    input: 1-D arr with int vals -- if not int vals, will raise an error
    output: m (ndarray): one-hot encoded matrix
            d (dict): also returns a dictionary original_val -> column in encoded matrix
    """
    for k in in_arr:
        if not isinstance(k, (int, np.integer)):
            print(str(type(k)))
            print("************* ERROR: Input arr does not have integer types")
            return None

    in_arr = np.array(in_arr, dtype=int)
    assert len(in_arr.shape) == 1  # no column, means it was a 1-D arr
    attr_vals_uniq_sorted = sorted(list(set(in_arr)))
    num_uniq_vals = len(attr_vals_uniq_sorted)
    if (num_uniq_vals == 2) and (attr_vals_uniq_sorted[0] == 0 and attr_vals_uniq_sorted[1] == 1):
        return in_arr, None

    index_dict = {}  # value to the column number
    for i in range(0, len(attr_vals_uniq_sorted)):
        val = attr_vals_uniq_sorted[i]
        index_dict[val] = i

    out_arr = []
    for i in range(0, len(in_arr)):
        tup = np.zeros(num_uniq_vals)
        val = in_arr[i]
        ind = index_dict[val]
        tup[ind] = 1  # set that value of tuple to 1
        out_arr.append(tup)

    return np.array(out_arr), index_dict

def check_data_file(fname):
    files = os.listdir(".")
    print("Looking for file '%s' in the current directory..." % fname)

    if fname not in files:
        print("'%s' not found! Downloading from UCI Archive..." % fname)
        addr = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/%s" % fname
        response = urllib.request.urlopen(addr)
        data = response.read().decode('utf-8')
        with open(fname, "w") as fileOut:
            fileOut.write(data)
        print("'%s' downloaded and saved locally.." % fname)
    else:
        print("File found in current directory..")

    print()
    return

def load_adult_data(load_data_size=None):
    attrs = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country']
    int_attrs = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week']
    sensitive_attrs = ['sex']
    attrs_to_ignore = ['sex', 'race', 'fnlwgt']
    attrs_for_classification = set(attrs) - set(attrs_to_ignore)

    data_files = ["adult.data", "adult.test"]

    X = []
    y = []
    x_control = {}

    attrs_to_vals = {}
    for k in attrs:
        if k in sensitive_attrs:
            x_control[k] = []
        elif k in attrs_to_ignore:
            pass
        else:
            attrs_to_vals[k] = []

    for f in data_files:
        check_data_file(f)

        for line in open(f):
            line = line.strip()
            if line == "":
                continue
            line = line.split(", ")
            if len(line) != 15 or "?" in line:
                continue

            class_label = line[-1]
            if class_label in ["<=50K.", "<=50K"]:
                class_label = -1
            elif class_label in [">50K.", ">50K"]:
                class_label = +1
            else:
                raise Exception("Invalid class label value")

            y.append(class_label)

            for i in range(0, len(line) - 1):
                attr_name = attrs[i]
                attr_val = line[i]
                if attr_name == "native_country":
                    if attr_val != "United-States":
                        attr_val = "Non-United-States"
                elif attr_name == "education":
                    if attr_val in ["Preschool", "1st-4th", "5th-6th", "7th-8th"]:
                        attr_val = "prim-middle-school"
                    elif attr_val in ["9th", "10th", "11th", "12th"]:
                        attr_val = "high-school"

                if attr_name in sensitive_attrs:
                    x_control[attr_name].append(attr_val)
                elif attr_name in attrs_to_ignore:
                    pass
                else:
                    attrs_to_vals[attr_name].append(attr_val)

    def convert_attrs_to_ints(d):
        for attr_name, attr_vals in d.items():
            if attr_name in int_attrs:
                continue
            uniq_vals = sorted(list(set(attr_vals)))

            val_dict = {}
            for i, val in enumerate(uniq_vals):
                val_dict[val] = i

            for i in range(0, len(attr_vals)):
                attr_vals[i] = val_dict[attr_vals[i]]
            d[attr_name] = attr_vals

    convert_attrs_to_ints(x_control)
    convert_attrs_to_ints(attrs_to_vals)

    for attr_name in attrs_for_classification:
        attr_vals = attrs_to_vals[attr_name]
        if attr_name in int_attrs or attr_name == "native_country":
            X.append(attr_vals)
        else:
            attr_vals, index_dict = get_one_hot_encoding(attr_vals)
            for inner_col in attr_vals.T:
                X.append(inner_col)

    X = np.array(X, dtype=float).T
    y = np.array(y, dtype=float)
    for k, v in x_control.items():
        x_control[k] = np.array(v, dtype=float)

    perm = list(range(0, len(y)))  # shuffle 대상인 range 객체를 list로 변환
    shuffle(perm)
    X = X[perm]
    y = y[perm]
    for k in x_control.keys():
        x_control[k] = x_control[k][perm]

    if load_data_size is not None:
        print("Loading only %d examples from the data" % load_data_size)
        X = X[:load_data_size]
        y = y[:load_data_size]
        for k in x_control.keys():
            x_control[k] = x_control[k][:load_data_size]

    return X, y, x_control

# 데이터를 로드하고 결과를 출력하는 코드 추가
if __name__ == "__main__":
    X, y, x_control = load_adult_data(load_data_size=1000)  # 데이터 로드 (필요에 따라 사이즈 조정 가능)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Sensitive attribute 'sex' distribution:", np.bincount(x_control['sex'].astype(int)))