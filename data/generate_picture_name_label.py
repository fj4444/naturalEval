import os
import ipdb
import random

def readDir(dirPath):
    if dirPath[-1] == '/':
        print(u'could not append / in an directory')
        return
    allFiles = []
    if os.path.isdir(dirPath):
        fileList = os.listdir(dirPath)
        for f in fileList:
            if os.path.splitext(f)[1] != ".png":
                pass
            else:
                f = dirPath+'/'+f
                if os.path.isdir(f):
                    subFiles = readDir(f)
                    allFiles = subFiles + allFiles #合并当前目录与子目录的所有文件路径
                else:
                    allFiles.append(f)
        return allFiles
    else:
        return 'Error,not a dir'
def generate_name():
    read_list = readDir("/root/zsn/natural/self_imp/actionclip/output")
    # ipdb.set_trace()
    with open("picture_name.txt", "w") as f:
        for i in read_list:
            # ipdb.set_trace()
            f.write(str(i) + "\n")
def generate_label():
    with open("picture_label.txt", "w") as f_label:
        for i in range(8304):
            f_label.write(str(round(random.random() * 7 - 0.5)) + "\n")
generate_label()