from prettytable import PrettyTable
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import numpy as np
import time
import os.path as op

from datasets import build_dataloader
from processor.processor import do_inference
from utils.checkpoint import Checkpointer
from utils.logger import setup_logger
from model import build_model
from utils.metrics import Evaluator
import argparse
from utils.iotools import load_train_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="IRRA Test")
    parser.add_argument("--config_file", default=r'./logs/Crop-Retrieval/20261015_092611_Mask2Cal_FullModel_Seed2_0.9MLM/configs.yaml')
    args = parser.parse_args()
    args = load_train_configs(args.config_file)

    args.training = False
    logger = setup_logger('IRRA', save_dir=args.output_dir, if_train=args.training)
    logger.info(args)
    device = "cuda"

    test_img_loader, test_txt_loader, num_classes = build_dataloader(args)
    logger.info(num_classes)
    model = build_model(args, num_classes=num_classes)
    checkpointer = Checkpointer(model)
    checkpointer.load(f=op.join(args.output_dir, 'best.pth'))
    # checkpointer.load(r"D:\Coding\MSFE-code\logs\Farm-Retrieval\20250923_221106_BDS_DDR\best.pth")
    model.to(device)
    do_inference(model, test_img_loader, test_txt_loader)