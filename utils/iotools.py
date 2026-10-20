# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
from PIL import Image, ImageFile
import errno
import json
import pickle as pkl
import os
import os.path as osp
import yaml
from easydict import EasyDict as edict
#允许读取“被截断 / 不完整”的图片
ImageFile.LOAD_TRUNCATED_IMAGES = True

#在高并发 I/O 或坏盘环境下，反复尝试读取图片，直到成功
def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img

#安全创建目录（多进程 / 重复调用不报错）
def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

#检查文件是否存在，不存在只警告，不中断程序
def check_isfile(path):
    isfile = osp.isfile(path)
    if not isfile:
        print("=> Warning: no file found at '{}' (ignored)".format(path))
    return isfile


def read_json(fpath):
    with open(fpath, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir_if_missing(osp.dirname(fpath))
    with open(fpath, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def get_text_embedding(path, length):
    with open(path, 'rb') as f:
        word_frequency = pkl.load(f)

#把 argparse 的参数保存为 YAML，方便复现实验
def save_train_configs(path, args):
    if not os.path.exists(path):
        os.makedirs(path)
    with open(f'{path}/configs.yaml', 'w') as f:
        yaml.dump(vars(args), f, default_flow_style=False)

#从 YAML 读回 dict 转成 EasyDict
def load_train_configs(path):
    with open(path, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    return edict(args)