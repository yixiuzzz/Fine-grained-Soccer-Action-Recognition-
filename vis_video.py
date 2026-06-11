import cv2
import torch
import torchvision.transforms as transforms
import random
import numpy as np
import copy
import tqdm
import torch.nn.functional as F
import os

from model.X3D import X3D

from utils.io import extract_label_txt



class Predictor(object):

    def __init__(self, 
                 pretrained_path: str, 
                 num_classes: int, 
                 clip_length: int,
                 model_size: str = 'm',
                 data_size: tuple = [224, 224],
                 torch_pretrained: bool = False,
                 device: str='cuda:0'):
        
        ckpt = torch.load(pretrained_path, map_location='cpu')
        self.model = X3D(num_classes=num_classes, 
                         model_size=model_size, 
                         torch_pretrained=torch_pretrained)
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device)
        self.model.eval()

        self.data_size = data_size

        self.normalize = transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225])
       
        self.clip_length = clip_length
        self.players = {}
        """
        'player_id' : player obj
        """

    def preprocess_player(self, arr: np.ndarray): #square
        arr = arr.astype(np.float32)
        t = torch.from_numpy(arr)
        t = t / 255.0
        t = t.permute(2, 0, 1) # H, W, C -> C, H, W

        # resize function
        C, H, W = t.size()
        if H == 0 or W == 0:
            return None

        target_size = [224, 224]
        resize_func = transforms.Resize(target_size, antialias=True)
        t = resize_func(t)

        t = self.normalize(t)

        return t


    @torch.no_grad()
    def process_frame(self, frame_id: int, frame: np.ndarray, label: dict):
        # frame: entir frame
        # label: this frame's label
        # 'player_id':
        #    {
        #        'bbox': player_bbox,
        #        'bbox_conf': conf,
        #        'label': lb_cls,
        #        'label_name': lb_name
        #    }

        for player_id in label:
            if player_id not in self.players:
                self.players[player_id] = Player(player_id, self.clip_length)

            # crop player from frame
            x0, y0, w, h = label[player_id]['bbox']
            x1, y1 = x0 + w, y0 + h
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            # player locations
            self.players[player_id].player_locs.append([x0, y0, x1, y1])

            # player frames
            player_frame = copy.deepcopy(frame)
            player_frame = player_frame[y0:y1, x0:x1]
            player_t = self.preprocess_player(player_frame)
            if player_t is None:
                continue
            
            self.players[player_id].player_clip.append(player_t)
            self.players[player_id].player_locs.append([x0, y0, x1, y1])
            # predict
            self.players[player_id].predict(frame_id, self.model)


        for player_id in label:
            self.players[player_id].plot(frame)



class Player(object):

    def __init__(self, player_id, clip_length: int, device:str = 'cuda:0'):
        self.clip_length = clip_length
        self.player_id = player_id
        self.player_locs = []
        self.player_clip = [] # store tensors
        self.device = device
        self.player_color = [random.randint(0, 255) for _ in range(3)]
        self.last_frame = -100
        self.this_pred = None
        self.conf = None

        # please replace this paramters by your situation
        self.cls2name = {
            '0' : 'move',
            '1' : 'idle',
            '2' : 'kick',
            '3' : 'fall'
        }
        self.cls2color = {
            'move': (0, 255, 0),  # 綠色
            'idle': (0, 255, 255),  #黃色
            'kick': (0, 0, 255),  # 紅色
            'fall': (255, 0, 0) # 藍色
        }


    @torch.no_grad()
    def predict(self, frame_id, model: torch.nn.Module):

        if frame_id - self.last_frame != 1:
            self.this_pred = None
            self.conf = None
            self.player_clip = []

        self.last_frame = frame_id

        if len(self.player_clip) < self.clip_length:
            self.this_pred = None
            self.conf = None
            return None

        data = torch.stack(self.player_clip)
        data = data.transpose(0, 1)
        data = data.to(self.device)
        data = data.unsqueeze(0)

        out = model(data)
        pred = torch.max(out, dim=-1)[1]

        # confident score
        values = F.softmax(out, dim=-1)
        confidence = values.gather(1, pred.unsqueeze(1)).squeeze(1)  

        # values = F.softmax(out, dim=-1)[0]
        # confident = torch.max(values)

        self.this_pred = self.cls2name[str(pred.cpu().item())]
        self.conf = float(confidence.cpu().item())

        del self.player_clip[0]


    def plot(self, arr: np.ndarray):
        if self.this_pred == 'none':

            self.this_pred = None
            self.conf = None
            return

        if self.this_pred == 'kick' and self.conf < 0.7:
            self.this_pred = 'move'

        # plot by last frame
        if self.this_pred is not None:
        # if self.this_pred is not None and self.conf > 0.5:
            x0, y0, x1, y1 = self.player_locs[-1]

            color = self.cls2color[str(self.this_pred)]

            cv2.rectangle(arr, (x0, y0), (x1, y1), color, 1)

            msg = f"{self.player_id} : {self.this_pred}"
            cv2.putText(arr, msg, (x0, y0 - 5), 1, 2, color, 2, 1)
            #conf
            cv2.putText(arr, f"{self.conf:.2f}", (x0, y0 - 30), 1, 2, color, 2, 1)

            self.this_pred = None
            self.conf = None





if __name__ == "__main__":


    test_game_index = 9 # folder name
    test_game = f'./{test_game_index}/' 

    pretrained_path = './records/X3D_fusion/X3D_16/backup/best.pt'
    save_root = f'./video/{os.path.basename(os.path.dirname(os.path.dirname(pretrained_path)))}/'
    if not os.path.isdir(save_root):
        os.makedirs(save_root)  

    for folder in os.listdir(test_game): 
        folder_path = os.path.join(test_game, folder)

        if os.path.isdir(folder_path):
            test_number = folder

            # test_number = 1

            test_label_path = test_game + f'{test_number}.txt'
            test_img_root = test_game + f'{test_number}/'

            labels = extract_label_txt(test_label_path)

            predictor = Predictor(pretrained_path = pretrained_path,
                                num_classes = 4, 
                                clip_length = 8,
                                model_size = 'm',
                                data_size = [224, 224],
                                torch_pretrained = False,
                                device = 'cuda:0')

            max_frame_id = max([int(x) for x in labels])

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # save_root
            # video_writer = cv2.VideoWriter(f'./video/v22/{test_game_index}-{test_number}.mp4', fourcc, 10.0, [1280, 720])
            video_writer = cv2.VideoWriter(f'{save_root}{test_game_index}-{test_number}.mp4', fourcc, 10.0, (1280, 720))
            
            pbar = tqdm.tqdm(total=max_frame_id, ascii=True)

            for frame_id in range(max_frame_id + 1):
                frame_id = str(frame_id)
                img_path = test_img_root + f'{frame_id}.jpg'
                arr = cv2.imread(img_path)

                predictor.process_frame(int(frame_id), arr, labels[frame_id])

                # show frame
                cv2.putText(arr, f'frame: {frame_id}', (10, 30), 1, 2, (0, 0, 255), 2, 1)
                # cv2.namedWindow('frame', 0)
                # cv2.imshow('frame', arr)
                cv2.waitKey(33)
                
                video_writer.write(arr)
                pbar.update(1)

            video_writer.release()

            pbar.close()

