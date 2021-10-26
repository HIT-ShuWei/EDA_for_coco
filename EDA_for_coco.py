import json
import matplotlib.pyplot as plt
import cv2 as cv
import os
import math
import numpy as np
import copy 
from numpy.core.fromnumeric import resize

class Analysis():
    '''
    self.x,y,w,h_list：数据集对应的归一化点的坐标/尺寸
    self.cls_frep:数据集中不同类别目标出现的频次
    self.img_source:不同来源的图片各自的数目

    '''
    def __init__(self, json_path):
        self.path = json_path
        self.id2file = {'start':'none'}   # id与img_name的对应表，用于反向查找
        self.img_source = {'download':0,'photoed':0,'generated':0}      # 统计图片来源

        self._get_json(self.path)
        self._gen_source(self.img_source, id2file=self.id2file)
        
        self.classes = self._generate_class_from_json()   # 数据集中按照顺序排列的cls名称
        

        self.cls_frq = {}   # 不同类别出现频率统计
        self.x_list, self.y_list, self.w_list, self.h_list = [], [], [], [] # 归一化坐标/尺寸统计
        _ = self._get_norm_xywh_and_freq(self.x_list, self.y_list, self.w_list, self.h_list, self.cls_frq)


    def _get_json(self, filename):
        f = open(filename)
        content = f.read()
        dataset = json.loads(content)
        self.images = dataset['images']
        self.annotations = dataset['annotations']
        self.categories = dataset['categories']
        f.close()

    def _generate_class_from_json(self):
        '''
        生成标签按照顺序排列的列表
        '''
        classes = []
        for id, category in enumerate(self.categories):
            classes.append(category['name'])
        return list(classes)

    def _gen_source(self, img_source, id2file=None, cls_idx=None):
        '''
        统计图片来源，同时建立id与图片的对应关系
        '''
        if cls_idx:
            # 如果有cls_idx，说明要统计制定的类别的图片来源
            for i, annotation in enumerate(self.annotations):
                if annotation["category_id"] != cls_idx:
                    continue
                else:
                    img_id = annotation['image_id']
                    file_name = self.id2file[img_id]
                    if 'Image' in file_name:
                        # Unity3D生成的图片
                        img_source['generated'] += 1
                    elif 'mixed' in file_name:
                        # 自己照的图片
                        img_source['photoed'] += 1
                    else:
                        # 网络下载的图片
                        img_source['download'] += 1
        else:
            for i, image in enumerate(self.images):
                if id2file:
                    # 在init时，建立id2file，其他时候不用
                    # 建立id：filename的字典
                    id2file[image['id']] = image['file_name']
                # 统计图片来源
                if 'Image' in image['file_name']:
                    # Unity3D生成的图片
                    img_source['generated'] += 1
                elif 'mixed' in image['file_name']:
                    # 自己照的图片
                    img_source['photoed'] += 1
                else:
                    # 网络下载的图片
                    img_source['download'] += 1
        

    def _get_wh_from_images(self,file_name):
        '''
        通过图片名称找到图片对应的尺寸
        '''
        for i, image in enumerate(self.images):
            if image['file_name'] == file_name:
                return (image['width'], image['height'])
            else:
                continue
            
    def _get_norm_xywh_and_freq(self,x_list, y_list, w_list, h_list, cls_frq, cls_idx=None):
        '''
        cls表示特别制定某一类别的统计，默认为None
        '''
        for i, annotation in enumerate(self.annotations):
            if cls_idx and cls_idx != annotation["category_id"]:
                continue
            # annotations中bbox是xywh保存的,xy为左上角坐标
            file_name = self.id2file[annotation['image_id']]
            # 获取对应图片的尺寸
            w_size,h_size = self._get_wh_from_images(file_name)
            # 获取bbox框的对应尺寸
            [x, y, w, h] = annotation['bbox']
            # 归一化处理
            x_list.append(x/w_size)
            y_list.append(y/h_size)
            w_list.append(w/w_size)
            h_list.append(h/h_size)
            # 获取标签
            label = self.classes[annotation['category_id']-1]
            if label not in cls_frq.keys():
                cls_frq[label] = 1
            else:
                cls_frq[label] += 1

        return x_list, y_list, w_list, h_list, cls_frq


    def analysis_single_class(self, cls):
        '''
        统计某一个类别的xywh分布、图片来源信息
        '''
        cls_idx = self.classes.index(cls)+1             #制定类别在列表中的idx
        x_list, y_list, w_list, h_list = [], [], [], [] #归一化分布
        cls_frq = {}    #统计类别出现频率
        self._get_norm_xywh_and_freq(x_list, y_list, w_list, h_list, cls_frq, cls_idx=cls_idx)

        img_source = {'download':0,'photoed':0,'generated':0}
        self._gen_source(img_source, cls_idx=cls_idx)


        return [x_list, y_list, w_list, h_list], cls_frq, img_source
    
    def visulize_single_class(self, img_path, cls, save_path=None):
        '''
        可视化展示目标类别的图片
        save_path存储结果，如果没有，则默认不存储
        '''
        cls_idx = self.classes.index(cls)+1  # 制定类别在列表中的idx
        img_names = []                       # 用于存储含有目标cls的图片名称
        bbox = []                            # 用于存储目标cls的位置，与file_name顺序对应
        for i, annotation in enumerate(self.annotations):
            if annotation['category_id'] != cls_idx:
                # 筛选目标cls的标签
                continue
            img_id = annotation['image_id']  # 当前标签对应的img-id 
            img_names.append(self.id2file[img_id])  # 添加当前bbox的图片名称
            bbox.append(annotation['bbox'])         # 添加对应bbox
        
        # 拼接图片前的参数设置
        bbox_per_img = 100              # 一张图片中展示的图片个数
        row_total, col_total = 10, 10   # 一张图片拼接的bbox行、列数
        result_list = []                # 存储最后结果的拼接图片

        # 开始拼接图片
        for i, img_name in enumerate(img_names):
            if i % bbox_per_img == 0:
                # 说明这张图片已经满了，需要新建一个图片
                if i != 0:
                    result_list.append(copy.deepcopy(img_empty))
                img_empty = np.ones((1000,1500,3), np.uint8)      # w=1500 h=1000
                row, col = 0, 0 #坐标重置
            file_name = os.path.join(img_path, img_name)    #对应图片路径
            img = cv.imread(file_name)      # 读取图片
            
            b,g,r = cv.split(img)
            img = cv.merge([r,g,b])         # 变成RGB图片
            [x,y,w,h] = bbox[i]             # 读取坐标框
            img = img[y:y+h,x:x+w]                # 截取bbox部分
            img = cv.resize(img,(150,100))        # 整合成统一的尺寸

            # 拼接
            x_offset = col * 150
            y_offset = row * 100
            
            img_empty[0+y_offset:100+y_offset, 0+x_offset:150+x_offset,] = img

            # 更新坐标
            if col < col_total-1 :
                col += 1
            elif col >= col_total-1 and row < row_total-1:
                col = 0
                row += 1
            else:
                continue
        result_list.append(copy.deepcopy(img_empty))
        # 绘图
        for i, result in enumerate(result_list):
            plt.figure(i)
            plt.imshow(result)
            if save_path:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                cv.imwrite(os.path.join(save_path,'{}_{}.jpg'.format(cls,i)), cv.cvtColor(result, cv.COLOR_RGB2BGR))




def show_cls_in_order(tgt_dict):
    '''
    将目标字典中元素按照从大到小的顺序用条形图呈现出来
    '''
    
    tgt_tuple_sort = sorted(tgt_dict.items(), key=lambda x: x[1], reverse=True)  # 按照alue排序成tuple
    tgt_dict = dict(tgt_tuple_sort) # 转换成字典
    num_category = len(tgt_dict)    # 元素个数
    
    plt.figure(figsize=(45,10))
    plt.bar(range(num_category), tgt_dict.values(), width=0.8, )
    plt.xticks(range(num_category), tgt_dict.keys())
    # plt.xlim(0, 50)     # x轴取值范围
    # plt.ylim(0, 50)     # y轴取值范围

    plt.ylabel("num")
    plt.xlabel("category")

