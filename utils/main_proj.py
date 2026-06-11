import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import random

from utils.utils import get_args
from utils.utils import get_schedule
from utils.utils import adjust_lr
from utils.utils import get_lr
from utils.utils import dict_to_str
from utils.utils import eval_this_epoch
from utils.utils import visualized
from utils.utils import set_seed
from model.focalloss import FocalLoss
from model.X3D_proj import X3D
from data.dataset import VideoDataset
from data.dataset import print_dict


def set_environment(args):
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if args.use_wandb:
        wandb.init(entity=args.wandb_entity,
                   project=args.project_name,
                   name=args.exp_name,
                   config=args)
        wandb.run.summary["best_acc"] = 0.0
        wandb.run.summary["best_epoch"] = 0

    # dataloader
    train_set = VideoDataset(
                 balance = True,
                 trans = True,
                 data_root = args.train_root, 
                 clip_length = args.clip_length,
                 data_size = args.data_size, # [H, W] 
        )
    train_loader = torch.utils.data.DataLoader(train_set, 
                                               num_workers=args.num_workers, 
                                               batch_size=args.batch_size,
                                               shuffle=True)

    test_set = VideoDataset(
                 balance = False,
                 trans = False,
                 data_root = args.test_root, 
                 clip_length = args.clip_length,
                 data_size = args.data_size, # [H, W]
        )
    test_loader = torch.utils.data.DataLoader(test_set, 
                                              num_workers=args.num_workers, 
                                              batch_size=args.batch_size,
                                              shuffle=False)

    print(f'Dataset Info:')
    print(f'    [train] sample: {len(train_set)} , batch: ({len(train_loader)})')
    print(f'    [test] sample: {len(test_set)} , batch: ({len(test_loader)})')
    print()


    # create model 
    # X3D
    model = X3D(num_classes=args.num_classes, 
                model_size=args.model_size, 
                torch_pretrained=args.torch_pretrained)
    model.to(args.device)

    # optimizer and lr_schedule
    optimizer = torch.optim.SGD(model.parameters(), 
                                lr=args.lr_schedule['max_value'],
                                nesterov=True,  
                                momentum=args.momentum,
                                weight_decay=args.wdecay)

    lr_schedule = get_schedule(schedule_info=args.lr_schedule, 
                               max_epochs=args.max_epochs, 
                               train_batchs=len(train_loader), 
                               warmup_batchs=args.warmup_batchs)

    return model, optimizer, train_loader, test_loader, lr_schedule


def compute_cosine_similarity(features):
    """
    计算 batch 内所有样本的 Cosine 相似度矩阵
    :param features: (B, C) 归一化后的特征
    :return: (B, B) 相似度矩阵
    """
    similarity_matrix = torch.matmul(features, features.T)  # (B, B)
    return similarity_matrix

def compute_relation_matrix(labels, num_classes):
    """
    计算 batch 内样本的关系矩阵 (Ground Truth)
    :param labels: (B,) 样本类别
    :param num_classes: 总类别数
    :return: (B, B) 关系矩阵
    """
    B = labels.size(0)
    one_hot_labels = F.one_hot(labels, num_classes).float()  # (B, N)
    relation_matrix = torch.matmul(one_hot_labels, one_hot_labels.T)  # (B, B)
    return relation_matrix

def consistency_loss(similarity_matrix, relation_matrix, temperature=0.5):
    """
    计算一致性损失 (Consistency Loss)
    :param similarity_matrix: (B, B) Cosine 相似度矩阵
    :param relation_matrix: (B, B) 关系矩阵 (1=同类, 0=不同类)
    :param temperature: 控制 softmax 范围
    :return: consistency loss
    """
    logits = similarity_matrix / temperature  # 调整温度系数
    loss = F.cross_entropy(logits, relation_matrix.float())  # 计算交叉熵损失
    return loss


def temporal_masking_fixed_T(data, min_keep=8, max_keep=16):
    B, C, T, H, W = data.shape
    assert T == 16, "This function assumes T=16 as input."

    masked_data = data.clone()  
    mask = torch.ones((B, 1, T, 1, 1), device=data.device) 

    for i in range(B):  
        keep_length = random.randint(min_keep, max_keep) # 8~16
        start_idx = random.randint(0, T - keep_length)
        masked_indices = list(range(start_idx)) + list(range(start_idx + keep_length, T))

        masked_data[i, :, masked_indices, :, :] = 0  
        mask[i, :, masked_indices, :, :] = 0  

    return masked_data, mask

def split_temporal_segments(data):
    """
    隨機取兩個 clip，長度在 [8,16] 之間，並用 Padding 填充到 T=16。
    Clip 皆從 0 開始，不重疊。
    
    :param data: [B, C, T, H, W]，T 固定為 16
    :return: (clip1, clip2, mask1, mask2)
    """
    B, C, T, H, W = data.shape
    assert T == 16, "This function assumes T=16 as input."

    # 初始化 clip 和 mask
    clip1 = torch.zeros_like(data)  # [B, C, T, H, W]，填充 0
    clip2 = torch.zeros_like(data)  
    mask1 = torch.zeros((B, 1, T, 1, 1), device=data.device)  # Mask 初始化
    mask2 = torch.zeros((B, 1, T, 1, 1), device=data.device)  

    for i in range(B):
        # 隨機決定 L1 和 L2，範圍 8~16
        L1 = random.randint(8, 16)
        L2 = random.randint(8, 16)

        # 取 L1 和 L2 的片段，從索引 0 開始
        clip1[i, :, :L1, :, :] = data[i, :, :L1, :, :]
        clip2[i, :, :L2, :, :] = data[i, :, :L2, :, :]

        # 產生 Mask（有效部分 = 1，Padding = 0）
        mask1[i, :, :L1, :, :] = 1
        mask2[i, :, :L2, :, :] = 1

    return clip1, clip2, mask1, mask2



def train(epoch, args,  model, optimizer, train_loader, lr_schedule):
    model.train()

    criterion = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5) # loss function
    n_train_samples = len(train_loader.dataset)
    pbar = tqdm.tqdm(total=n_train_samples, ascii=True)
   
    for it, (data, label) in enumerate(train_loader):

        # adjust learning rate
        n_iters = len(train_loader) * epoch + it 
        adjust_lr(n_iters, lr_schedule, optimizer)

        # forward 
        data, label = data.to(args.device), label.to(args.device)

        # data, mask = temporal_masking_fixed_T(data)

        # if (epoch % 10 == 0) and it == 0:
        #     visualized(data, epoch, args, device=args.device)
        # exit()

        # logits, features = model(data, mask) # [batch_size, num_classes]


        clip1, clip2, mask1, mask2 = split_temporal_segments(data)

        # **模型 Forward**
        logits1, features1 = model(clip1, mask1)  # 第一段影片
        logits2, features2 = model(clip2, mask2)  # 第二段影片


        # calculate loss
        onehot_label = nn.functional.one_hot(label, num_classes=args.num_classes)
        onehot_label = onehot_label.float()
        # class_loss = criterion(logits, onehot_label)

        class_loss1 = criterion(logits1, onehot_label)
        class_loss2 = criterion(logits2, onehot_label)
        class_loss = 0.5 * (class_loss1 + class_loss2)  # 兩段影片的分類 Loss 平均

        # mask
        # mask_weight = mask.squeeze(1).squeeze(-1).squeeze(-1).mean(dim=1)  # [B]
        # class_loss = (class_loss * mask_weight).mean()  

        # consistency loss
        # similarity_matrix = compute_cosine_similarity(features)
        # relation_matrix = compute_relation_matrix(label, args.num_classes)
        # con_loss = consistency_loss(similarity_matrix, relation_matrix)

        # total loss
        # loss = class_loss + 0.5 * con_loss
        loss = class_loss

        # backward
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        # show training progress
        with torch.no_grad():
            if (it + 1) % args.log_freq == 0: # log_freq=5
                msg = {}
                # train accuracy
                pred = torch.max(logits, dim=1)[1]  # torch.max(value, indices) 
                correct = (pred == label).float().sum().cpu().item()
                acc = round(correct / label.size(0) * 100, 3) # label.size(0): sample number
                msg['train/acc'] = acc 
                msg['train/loss'] = loss.cpu().item()
                msg['train/lr'] = get_lr(optimizer)
                msg['run/epoch'] = epoch

                if args.use_wandb:
                    wandb.log(msg)
                    
                msg = dict_to_str(msg)
                pbar.set_description(msg)

        pbar.update(label.size(0))

    pbar.close()



@torch.no_grad()
def eval(args, epoch, model, test_loader):
    model.eval()

    test_loss = 0.0
    corrects = 0.0
    total = len(test_loader.dataset)

    conf_mat = np.zeros([args.num_classes, args.num_classes]) # 4x4

    pbar = tqdm.tqdm(total=len(test_loader.dataset), ascii=True)
    for data, label in test_loader:

        data, label = data.to(args.device), label.to(args.device)

        logits, _ = model(data) 

        loss = F.cross_entropy(logits, label)
        test_loss += loss * (label.size(0) / total) # get avg results
        pred = torch.max(logits, dim=1)[1]
        
        c = (pred == label).float().sum().cpu().item()
        corrects += c
        
        for i in range(label.size(0)):
            conf_mat[int(pred[i]), int(label[i])] += 1 

        pbar.set_description(f'testing acc: {round(corrects / total * 100, 4)}%')
        pbar.update(label.size(0))

    pbar.close()

    test_acc = corrects / total
    test_acc = round(test_acc * 100, 4)
    if args.use_wandb:
        msg = {}
        msg['test/acc'] = test_acc
        msg['test/loss'] = test_loss
        wandb.log(msg)

    # confusion matrix
    plt.cla()
    plt.clf()
    plt.matshow(conf_mat, cmap='Blues')
    for (i, j), val in np.ndenumerate(conf_mat):
        plt.text(j, i, f'{val:.0f}', ha='center', va='center', color='Black', fontsize=16)
 
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.gca().xaxis.set_label_position('top')
    plt.colorbar()
    
    plt.savefig(f'{args.save_dir}/conf_mat/ep{epoch}.jpg')

    return test_acc



def record_best_model(test_acc, best_acc, ckpt, args, epoch):
    if test_acc >= best_acc:
        best_acc = test_acc
        torch.save(ckpt, args.save_dir + '/backup/best.pt')

        if args.use_wandb:
            wandb.run.summary["best_acc"] = best_acc
            wandb.run.summary["best_epoch"] = epoch
            
    return max(test_acc, best_acc)



def main():
    args = get_args()
    set_seed(args.seed)
    model, optimizer, train_loader, test_loader, lr_schedule =\
        set_environment(args)

    best_acc = 0.0
    for epoch in range(args.max_epochs):
        train(epoch, args, model, optimizer, train_loader, lr_schedule)
        ckpt = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch':epoch}
        torch.save(ckpt, args.save_dir + '/backup/last.pt')

        if eval_this_epoch(epoch, args.eval_freq):
            test_acc = eval(args, epoch, model, test_loader)
            best_acc = record_best_model(test_acc, best_acc, ckpt, args, epoch)


if __name__ == "__main__":
    main()