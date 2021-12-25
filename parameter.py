# -*- coding: utf-8 -*-

ORI_DATA = "./ori_data"
ORI_TRAIN = "./ori_data/train"
ORI_VAL = "./ori_data/val"

GEN_TRAIN = "./gen_data/train"
GEN_VAL = "./gen_data/val"

GEN_RATE = 5

EPOCH_NUM = 1
BATCH_SIZE = 32
IMG_SIZE = 224
PRE_SHOW = 16

INDEX = {'00_00': 0, '00_01': 1, '00_02': 2, '00_03': 3, '00_04': 4, '00_05': 5, '00_06': 6, '00_07': 7,
         '00_08': 8, '00_09': 9, '01_00': 10, '01_01': 11, '01_02': 12, '01_03': 13, '01_04': 14,
         '01_05': 15, '01_06': 16, '01_07': 17, '02_00': 18, '02_01': 19, '02_02': 20, '02_03': 21,
         '03_00': 22, '03_01': 23, '03_02': 24, '03_03': 25}
INVERTED = {0: 'Plastic Bottle', 1: 'Hats', 2: 'Newspaper', 3: 'Cans', 4: 'Glassware', 5: 'Glass Bottle',
            6: 'Cardboard', 7: 'Basketball', 8: 'Paper', 9: 'Metalware', 10: 'Disposable Chopsticks',
            11: 'Lighter', 12: 'Broom', 13: 'Old Mirror', 14: 'Toothbrush', 15: 'Dirty Cloth', 16: 'Seashell',
            17: 'Ceramic Bowl', 18: 'Paint bucket', 19: 'Battery', 20: 'Fluorescent lamp', 21: 'Tablet capsules',
            22: 'Orange Peel', 23: 'Vegetable Leaf', 24: 'Eggshell', 25: 'Banana Peel'}
RE_INVERTED = {'Plastic Bottle': 0, 'Hats': 1, 'Newspaper': 2, 'Cans': 3, 'Glassware': 4, 'Glass Bottle': 5,
               'Cardboard': 6, 'Basketball': 7, 'Paper': 8, 'Metalware': 9, 'Disposable Chopsticks': 10,
               'Lighter': 11, 'Broom': 12, 'Old Mirror': 13, 'Toothbrush': 14, 'Dirty Cloth': 15, 'Seashell': 16,
               'Ceramic Bowl': 17, 'Paint bucket': 18, 'Battery': 19, 'Fluorescent lamp': 20, 'Tablet capsules': 21,
               'Orange Peel': 22, 'Vegetable Leaf': 23, 'Eggshell': 24, 'Banana Peel': 25}

MODEL = "./results/"
LOG = "./results/tb_results/"
CLASS_NUM = 26
