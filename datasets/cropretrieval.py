import os.path as op
from typing import List

from utils.iotools import read_json
from .bases import BaseDataset


class CropRetrieval(BaseDataset):
    dataset_dir = 'Crop-Retrieval'

    def __init__(self, root='', verbose=True):
        super(CropRetrieval, self).__init__()
        self.dataset_dir = op.join(root, self.dataset_dir)
        self.img_dir = op.join(self.dataset_dir, 'imgs/')

        self.train_anno_path = op.join(self.dataset_dir, 'anno/train.json')
        self.test_anno_path = op.join(self.dataset_dir, 'anno/test.json')
        self.val_anno_path = op.join(self.dataset_dir, 'anno/val.json')
        self._check_before_run()

        # 直接加载各个分割的标注文件
        self.train_annos = read_json(self.train_anno_path)
        self.test_annos = read_json(self.test_anno_path)
        self.val_annos = read_json(self.val_anno_path)

        self.train, self.train_id_container = self._process_anno(self.train_annos, training=True)
        self.test, self.test_id_container = self._process_anno(self.test_annos)
        self.val, self.val_id_container = self._process_anno(self.val_annos)

        if verbose:
            self.logger.info("=> Crop-Retrieval Images and Captions are loaded")
            self.show_dataset_info()

    def _process_anno(self, annos: List[dict], training=False):
        pid_container = set()
        if training:
            dataset = []
            
            for anno in annos:
                pid = int(anno['image_id'])
                pid_container.add(pid)
                
                # 构建图像路径
                img_path = op.join(self.img_dir, anno['image'])
                
                # 使用数据文件中提供的image_id
                image_id = int(anno['image_id'])
                
                # 处理caption - 有些是字符串，有些是列表
                if isinstance(anno['caption'], str):
                    dataset.append((pid, image_id, img_path, anno['caption']))
                elif isinstance(anno['caption'], list):
                    for caption in anno['caption']:
                        dataset.append((pid, image_id, img_path, caption))
            
            return dataset, pid_container
        else:
            dataset = {}
            img_paths = []
            captions = []
            image_pids = []
            caption_pids = []
            image_ids = []
            
            for anno in annos:
                pid = int(anno['image_id'])
                pid_container.add(pid)
                
                img_path = op.join(self.img_dir, anno['image'])
                image_id = int(anno['image_id'])
                
                # 添加图像信息
                img_paths.append(img_path)
                image_pids.append(pid)
                image_ids.append(image_id)
                
                # 处理caption
                if isinstance(anno['caption'], str):
                    captions.append(anno['caption'])
                    caption_pids.append(pid)
                elif isinstance(anno['caption'], list):
                    for caption in anno['caption']:
                        captions.append(caption)
                        caption_pids.append(pid)
            
            dataset = {
                "image_pids": image_pids,
                "img_paths": img_paths,
                "image_ids": image_ids,
                "caption_pids": caption_pids,
                "captions": captions
            }
            return dataset, pid_container

    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not op.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not op.exists(self.img_dir):
            raise RuntimeError("'{}' is not available".format(self.img_dir))
        if not op.exists(self.train_anno_path):
            raise RuntimeError("'{}' is not available".format(self.train_anno_path))
        if not op.exists(self.test_anno_path):
            raise RuntimeError("'{}' is not available".format(self.test_anno_path))
        if not op.exists(self.val_anno_path):
            raise RuntimeError("'{}' is not available".format(self.val_anno_path))