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
from model.X3D import X3D
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

        # if (epoch % 10 == 0) and it == 0:
        #     visualized(data, epoch, args, device=args.device)

        # b, c, t, h, w = data.shape
        # select_length = 8
        
        # selected_samples = []
        # selected_indices_log = []  

        # for i in range(b):
        #     frame_indices = sorted(random.sample(range(t), select_length))  
        #     selected_indices_log.append(frame_indices)  
        #     selected_sample = data[i, :, frame_indices, :, :]  
        #     selected_samples.append(selected_sample)

        # selected_frames = torch.stack(selected_samples, dim=0)  # [B, C, 8, H, W]

        # print(f"Epoch {epoch}, Iter {it}:")
        # for a in range(b):
        #     print(f" Sample {a}: Selected frames {selected_indices_log[a]}")


        out = model(data) # [batch_size, num_classes]

        # calculate loss
        onehot_label = nn.functional.one_hot(label, num_classes=args.num_classes)
        onehot_label = onehot_label.float()
        loss = criterion(out, onehot_label)

        # backward
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

        # show training progress
        with torch.no_grad():
            if (it + 1) % args.log_freq == 0: # log_freq=5
                msg = {}
                # train accuracy
                pred = torch.max(out, dim=1)[1]  # torch.max(value, indices) 
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

        out = model(data) 

        loss = F.cross_entropy(out, label)
        test_loss += loss * (label.size(0) / total) # get avg results
        pred = torch.max(out, dim=1)[1]
        
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