import torch
import ipdb
import os, sys

data_dir = "/root/zsn/data/experiment_picture/"

input_name_list = []
output_name_list = []

for root, dirs, files in os.walk(data_dir, topdown=False):
    # ipdb.set_trace()
    for name in files:
        input_name_list.append(os.path.join(root, name))
        output_name_list.append(os.path.join(data_dir + root[25:], name[:-4] + ".png"))

file_dict = {}
for file in input_name_list:
    fileprefix = "/".join(file.split("/")[:-1])
    if fileprefix.endswith(".png"):
        if (fileprefix not in file_dict.keys()):
            file_dict[fileprefix] = []
        file_dict[fileprefix].append(file)

key_list = []
value_list = []
for item in file_dict.items():
    key_list.append(item[0])
    value_list.append(item[1])

# ipdb.set_trace()
split_01 = int(0.1 * len(key_list))
split_015 = int(0.15 * len(key_list))
split_02 = int(0.2 * len(key_list))

train_data_08, test_data_08, valid_data_08 = torch.utils.data.random_split(key_list, [len(key_list) - 2 * split_01, split_01, split_01])
train_data_07, test_data_07, valid_data_07 = torch.utils.data.random_split(key_list, [len(key_list) - 2 * split_015, split_015, split_015])
train_data_06, test_data_06, valid_data_06 = torch.utils.data.random_split(key_list, [len(key_list) - 2 * split_02, split_02, split_02])
file_list_name = ["train_data_08", "test_data_08", "valid_data_08", "train_data_07", "test_data_07", "valid_data_07", "train_data_06", "test_data_06", "valid_data_06"]
file_label_list = [train_data_08, test_data_08, valid_data_08, train_data_07, test_data_07, valid_data_07, train_data_06, test_data_06, valid_data_06]
for str, file in zip(file_list_name, file_label_list):
    with open("../../data/" + str + "_label.txt", "w") as f:
        for file_idx in range(len(file)):
            f.write(file.__getitem__(file_idx).split("experiment_picture/")[1] + "\n")