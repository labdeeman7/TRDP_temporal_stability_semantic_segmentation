import numpy as np
import os
from PIL import Image
import argparse
from cityscapesscripts.preparation.json2labelImg import createLabelImage

def dataset_size_reduction():
    raise NotImplementedError

def prepare_pseudo_label_gt(ann_dir):
    '''
    Change the pseudo-labels gotten from the efficient segmentation paper link, 
    from 3 dimensions to one dimension, 
    Convert to label_ID 
    change the name as well to *_label_id
    '''

    for path, subdirs, files in os.walk(ann_dir):
        for name in files:
            if name.endswith(".jpg") or name.endswith(".jpeg") or name.endswith("png"):
                file_path = os.path.join(path,name)
                ann_img = np.array(Image.open(file_path).convert('L'))
                label_tran_id_img = id_to_train_id(ann_img)
                label_tran_id_img = Image.fromarray(label_tran_id_img)
                file_path = file_path.replace("_leftImg8bit", "_gtFine_labelTrainIds")
                label_tran_id_img.save(file_path)

def id_to_train_id(img, ignore_label=255):
    id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                    3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                    7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                    14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                    18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                    28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}
    
    img_copy = img.copy()

    for k, v in id_to_trainid.items():
        img_copy[img == k] = v
    
    return img_copy


def main():
    parser = argparse.ArgumentParser(description='create labelTrainIds from annotations')
    parser.add_argument("--ann_dir", type=str, help="path to annotation_dir")

    args = parser.parse_args()
    ann_dir = args.ann_dir

    prepare_pseudo_label_gt(ann_dir)

if __name__ == '__main__':
    main()