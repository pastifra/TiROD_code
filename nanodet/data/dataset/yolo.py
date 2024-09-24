# Copyright 2023 cansik.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import fnmatch
import time
from collections import defaultdict
from typing import Optional, Sequence

import numpy as np
from imagesize import imagesize
from pycocotools.coco import COCO

from .coco import CocoDataset

def get_file_list(path, type):
    file_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            ext = os.path.splitext(apath)[1]
            if ext == type:
                file_names.append(apath)
    return file_names

class CocoYolo(COCO):
    def __init__(self, annotation):
        """
        Constructor of Microsoft COCO helper class for
        reading and visualizing annotations.
        :param annotation: annotation dict
        :return:
        """
        # load dataset
        super().__init__()
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        dataset = annotation
        assert type(dataset) == dict, "annotation file format {} not supported".format(
            type(dataset)
        )
        self.dataset = dataset
        self.createIndex()


class YoloDataset(CocoDataset):
    def __init__(self, class_names, **kwargs):
        self.class_names = class_names
        super(YoloDataset, self).__init__(**kwargs)

    """
    @staticmethod
    #This works for sequential factor analysis
    def _find_image(ann_to_match, path):

        image_prefix = ann_to_match[2].split('.')[0] + '.jpg'
        task = ann_to_match[0]
        class_name = ann_to_match[1]
        task_to_segment = {
            'Strong': ['segment1', 'segment2', 'segment3'],
            'Normal': ['segment4', 'segment5', 'segment6'],
            'Weak': ['segment7', 'segment8', 'segment9']
        }
        segments = task_to_segment.get(task)

        for segment in segments:
            segment_path = os.path.join(path, segment, class_name)
            for root, dirs, files in os.walk(segment_path):
                for file in files:
                    if fnmatch.fnmatch(file, image_prefix):
                        return os.path.join(root, file)
        return None
    """
    @staticmethod
    def _find_image(ann_to_match, path):

        image_prefix = ann_to_match[2].split('.')[0] + '.jpg'
        task = ann_to_match[0]
        class_name = ann_to_match[1]

        segment_path = os.path.join(path, class_name)
        for root, dirs, files in os.walk(segment_path):
            for file in files:
                if fnmatch.fnmatch(file, image_prefix):
                    return os.path.join(root, file)
        return None

    def yolo_to_coco(self, ann_path):
        """
        convert yolo annotations to coco_api
        :param ann_path:
        :return:
        """
        logging.info("loading annotations into memory...")
        tic = time.time()
        ann_file_names = get_file_list(ann_path, type=".txt")
        
        logging.info("Found {} annotation files.".format(len(ann_file_names)))
        image_info = []
        categories = []
        annotations = []
        for idx, supercat in enumerate(self.class_names):
            categories.append(
                {"supercategory": supercat, "id": idx + 1, "name": supercat}
            )
        logging.warning("WARNING! Keeping only annotations of these categories {}! ".format(categories))
        ann_id = 1

        for idx, txt_name in enumerate(ann_file_names):
            ann_file = os.path.join(ann_path, txt_name)
            
            # Split the path into a list of directories and the file
            parts = ann_file.split(os.sep)

            # Get the last two directories and the file
            ann_to_match = parts[-3:] # Output: ['Normal', 'paper_cutter_01', 'frame0094.txt']
            image_file = self._find_image(ann_to_match, self.img_path)
            
            if image_file is None:
                continue

            width, height = imagesize.get(image_file)
            image_file_parts = image_file.split(os.sep)

            file_name = os.sep.join(image_file_parts[-2:])#file_name = os.sep.join(image_file_parts[-3:]) #FOR SEQUENTIAL FACTOR ANALYSIS

            info = {
                "file_name": file_name,
                "height": height,
                "width": width,
                "id": idx + 1,
            }
            image_info.append(info)

            with open(ann_file, "r") as f:
                x_min = int(float(f.readline().strip()))
                y_min = int(float(f.readline().strip()))
                x_max = int(float(f.readline().strip()))
                y_max = int(float(f.readline().strip()))

            w = x_max - x_min
            h = y_max - y_min
            
            coco_box = [max(x_min, 0), max(y_min, 0), min(w, width), min(h, height)]

            class_name = ann_to_match[1]
            class_name, number = class_name.rsplit('_', 1)

            cat_id = self.class_names.index(class_name) + 1

            ann = {
                "image_id": idx + 1,
                "bbox": coco_box,
                "category_id": cat_id,
                "iscrowd": 0,
                "id": ann_id,
                "area": coco_box[2] * coco_box[3],
            }
            annotations.append(ann)
            ann_id += 1

        coco_dict = {
            "images": image_info,
            "categories": categories,
            "annotations": annotations,
        }

        #logging info
        print(
            "Load {} txt files and {} boxes".format(len(image_info), len(annotations))
        )
        logging.info("Done (t={:0.2f}s)".format(time.time() - tic))
        return coco_dict

    def get_data_info(self, ann_path):
        """
        Load basic information of dataset such as image path, label and so on.
        :param ann_path: coco json file path
        :return: image info:
        [{'file_name': '000000000139.jpg',
          'height': 426,
          'width': 640,
          'id': 139},
         ...
        ]
        """
        coco_dict = self.yolo_to_coco(ann_path)
        self.coco_api = CocoYolo(coco_dict)
        self.cat_ids = sorted(self.coco_api.getCatIds())
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cats = self.coco_api.loadCats(self.cat_ids)
        self.img_ids = sorted(self.coco_api.imgs.keys())
        img_info = self.coco_api.loadImgs(self.img_ids)
        return img_info
