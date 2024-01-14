'''
@Author  ：GBDJ
@Date    ：2023/7/5 16:21 
'''
# 更改后的数据集！

import glob
import os.path as osp
from typing import List
import os
from mmengine.dataset import BaseDataset

from mmrotate.registry import DATASETS


@DATASETS.register_module()
class FAIR1MDataset(BaseDataset):
    # FAIR1M-v1.0 数据集！
    # 类别定义！
    METAINFO = {
        'classes':
            ('Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship',
             'Dry Cargo Ship', 'Warship', 'other-ship', 'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck',
             'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor', 'other-vehicle', 'Boeing737',
             'Boeing747', 'Boeing777', 'Boeing787', 'ARJ21', 'C919', 'A220', 'A321', 'A330', 'A350',
             'other-airplane', 'Intersection', 'Roundabout', 'Bridge', 'Baseball Field', 'Basketball Court',
             'Football Field', 'Tennis Court'),
    # 可视化时用到的颜色列表！暂未实现！
    # 'palette': [(165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
    #             (138, 43, 226), (255, 128, 0), (255, 0, 255),
    #             (0, 255, 255), (255, 193, 193), (0, 51, 153),
    #             (255, 250, 205), (0, 139, 139), (255, 255, 0),
    #             (147, 116, 116), (0, 0, 255)]
    }
    def __init__(self,
                 diff_thr: int = 10,
                 shape_thr: float = 0.7,
                 img_suffix: str = 'tif',
                 **kwargs) -> None:
        """

        @param diff_thr: 难度阈值，低于该阈值的锚框会被忽略
        @param img_suffix: 后缀名（默认为.tif）
        @param kwargs:
        """
        self.diff_thr = diff_thr
        self.shape_thr = shape_thr
        self.img_suffix = img_suffix
        super().__init__(**kwargs)

    def load_data_list(self) -> List[dict]:
        """读取ann文件到self.ann_file
        Returns:
            List[dict]: 标签列表
        """
        cls_map = {c: i
                   for i, c in enumerate(self.metainfo['classes'])
                   }  # in mmdet v2.0 label is 0-based
        data_list = []
        if self.ann_file == '':
            # annfile为空，没有标签！
            img_files = glob.glob(
                osp.join(self.data_prefix['img_path'], f'*.{self.img_suffix}'))
            for img_path in img_files:
                data_info = {}
                data_info['img_path'] = img_path
                img_name = osp.split(img_path)[1]
                data_info['file_name'] = img_name
                img_id = img_name[:-4]
                data_info['img_id'] = img_id

                instance = dict(bbox=[], bbox_label=[], ignore_flag=0)
                data_info['instances'] = [instance]
                data_list.append(data_info)
            return data_list
        else:
            # ann单独放入一个文件夹的情况，根据标签找文件
            txt_files = glob.glob(osp.join(self.ann_file, '*.txt'))
            if len(txt_files) == 0:
                raise ValueError('There is no txt file in 'f'{self.ann_file}')
            for txt_file in txt_files:
                data_info = {}
                img_id = osp.split(txt_file)[1][:-4]
                data_info['img_id'] = img_id
                img_name = img_id + f'.{self.img_suffix}'
                data_info['file_name'] = img_name
                data_info['img_path'] = osp.join(self.data_prefix['img_path'],img_name)
                assert os.path.exists(data_info['img_path'])
                instances = []
                with open(txt_file) as f:
                    s = f.readlines()
                    for si in s:
                        instance = {}
                        bbox_info = si.strip().split(" ")
                        instance['bbox'] = [float(i) for i in bbox_info[:8]]
                        cls_name = " ".join(bbox_info[8:-2])
                        # assert cls_name in self.metainfo
                        instance['bbox_label'] = cls_map[cls_name]
                        instance['difficulty']=float(bbox_info[-2])
                        instance['shape_in_box']=float("-".join(bbox_info[-1].split("-")[1:]))
                        instance['serialnumber'] = bbox_info[-1]
                        instance['ignore_flag'] = 0
                        if instance['difficulty'] < self.diff_thr:
                            instance['ignore_flag'] = 1
                        if instance['shape_in_box']<self.shape_thr:
                            instance['ignore_flag'] = 1
                        instances.append(instance)
                data_info['instances'] = instances
                data_list.append(data_info)
            return data_list

    def filter_data(self) -> List[dict]:
        """
            过滤一些样本
        """
        # 测试模式需要过所有的图片
        if self.test_mode:
            return self.data_list
        # 清除空样本的图片
        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False) \
            if self.filter_cfg is not None else False

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            # 清除空样本的图片
            if filter_empty_gt and len(data_info['instances']) == 0:
                continue
            valid_data_infos.append(data_info)

        return valid_data_infos

    def get_cat_ids(self, idx: int) -> List[int]:
        """Get DOTA category ids by index.

        Args:
            idx (int): Index of data.
        Returns:
            List[int]: All categories in the image of specified index.
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]

@DATASETS.register_module()
class FAIR1Mv20Dataset(FAIR1MDataset):
    pass