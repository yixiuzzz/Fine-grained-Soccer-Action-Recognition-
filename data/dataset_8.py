import torch
import os
import torchvision.transforms as transforms
import cv2
import numpy as np
import random


def print_dict(data):
    for name in data:
        print(f'{name} : {data[name]}')


class DatasetTools(object):

    def get_video_datainfo(self, data_root, balance = True) -> list:
        
        data_infos = []
        class_counts = {}
        cls_folders = os.listdir(data_root) # train_data : 0, 1, 2, 3, none
        cls_folders = [x for x in cls_folders if x != 'none'] # skip none folder
        cls_folders.sort() # [0, 1, 2, 3]
        
        # 計算最大影片數量: max_n --> 平衡用
        for i in range(4):           
            class_counts[str(i)] = 0 # 初始化 class_counts = {'0': 0, '1': 0, '2': 0, '3': 0}

        for i, folder in enumerate(cls_folders): 

            if i == 0:
                label = 0
            elif i == 1:
                label = 1
            elif i == 2:
                label = 2
            elif i == 3:
                label = 3

            videos = os.listdir(data_root + '/' + folder + '/') # /train_data/0/0~... 影片資料夾

            for video in videos:
                class_counts[str(label)] += 1  # class_counts = {'0': 482, '1': 195 ...} 資料夾數量

        max_n = max([class_counts[name] for name in class_counts]) # [482, 195, ...] 


        # balance data
        for i, folder in enumerate(cls_folders):

            if i == 0:
                label = 0
            elif i == 1:
                label = 1
            elif i == 2:
                label = 2
            elif i == 3:
                label = 3

            videos = os.listdir(data_root + '/' + folder + '/') # /train_data/0/0~... 影片資料夾

            for video in videos: # data_infos
                tmp = {
                    'video_path': data_root + '/' + folder + '/' + video + '/',  # train_data/0/0/ (folder_path)
                    'label': label
                }

                if balance:
                    repeat_time = int(max_n / max(class_counts[str(label)], 1)) 
                else:
                    repeat_time = 1

                for _ in range(repeat_time):
                    data_infos.append(tmp) 

        return data_infos, class_counts

        # data_infos: tmp = {
        #             'video_path': data_root + '/' + folder + '/' + video + '/',  # train_data/0/0/ (image_folder_path)
        #             'label': label
        #         }

        
    def build_sequence_datainfo(self, 
                                video_data_infos: list, # data_infos, class_counts
                                clip_length: int) -> list: 
                        
        seq_data_infos = []
        class_counts = {} # just for analysis
        for video_info in video_data_infos: # {video_path, label}
            
            if video_info['label'] not in class_counts:
                class_counts[video_info['label']] = 0

            n_frames = len(os.listdir(video_info['video_path'])) 
            video_index = os.path.basename(os.path.normpath(video_info['video_path']))

            if n_frames < clip_length:
                continue

            elif clip_length <= n_frames < 16:
                tmp = {
                    'imgs': [], 
                    'label': video_info['label']
                }

                for i in range(8):  
                    tmp['imgs'].append(video_info['video_path'] + f'{i}.jpg')

                seq_data_infos.append(tmp)  
                class_counts[tmp['label']] += 1 

            else:
                max_clips = n_frames // 16
                frame_idx = list(range(max_clips * 16))
                start_pt = list(range(0, len(frame_idx), 16))

                for pt in start_pt:
                    tmp = {
                        'imgs': [], 
                        'label': video_info['label']
                    }

                    for i in range(pt, pt + 8):  
                        tmp['imgs'].append(video_info['video_path'] + f'{i}.jpg')

                    seq_data_infos.append(tmp)  
                    class_counts[tmp['label']] += 1 

        return seq_data_infos, class_counts


    def build_clip(self, data_info, data_size, trans):
        clip = []

        for img_path in data_info['imgs']:
            arr = cv2.imread(img_path) # Numpy array
            arr = arr.astype(np.float32) # unit8(0~255) --> folat32 
            t = torch.from_numpy(arr) # Numpy array --> PyTorch tesnor
            t = t / 255.0 
            t = t.permute(2, 0, 1) # H, W, C -> C, H, W
            
            # resize to square
            resize_func = transforms.Resize(data_size, antialias=True)
            t = resize_func(t)

            clip.append(t)
            
        clip = torch.stack(clip) #[num_frames, C, H, W]
        clip = trans(clip)
        
        return clip


class VideoDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 balance: bool,
                 trans: bool,
                 data_root: str, 
                 clip_length : int,  
                 data_size: tuple): 

        assert data_size[0] == data_size[1] # So for, only support square frames
        self.data_size = data_size
        self.clip_length = clip_length

        self.dtools = DatasetTools()

        self.trans = trans
        if trans:
            self.transforms = transforms.Compose([
                    transforms.ColorJitter(0.08, 0.08, 0.08, 0.02), # brightness, contrast, saturation, hue
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomGrayscale(),
                    # transforms.RandomAdjustSharpness(0.4),
                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ])
        else:
            self.transforms = transforms.Compose([
                    transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
                ])

        self.balance = balance
        
        self.video_data_infos, class_counts = self.dtools.get_video_datainfo(data_root, balance)

        self.seq_data_infos, class_counts = \
            self.dtools.build_sequence_datainfo(self.video_data_infos, clip_length)
        
        print_dict(class_counts)


    def __len__(self):
        return len(self.seq_data_infos)

        
    def __getitem__(self, index):
        info = self.seq_data_infos[index]
        data = self.dtools.build_clip(info, self.data_size, self.transforms)

        label = info['label']

        data = data.transpose(0, 1) # [T, C, H, W] -> [C, T, H, W]

        return data, label



        


