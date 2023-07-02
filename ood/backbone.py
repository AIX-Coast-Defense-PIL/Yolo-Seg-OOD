import os, sys
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

import sys 
work_path = os.path.join(os.getcwd(), os.pardir) if '/ood' in os.getcwd() else os.getcwd()
work_path = work_path if 'Yolo-Seg-OOD' in work_path else os.path.join(work_path, 'Yolo-Seg-OOD')
sys.path.append(work_path)

from ood.args_loader import get_args
from ood.data_loader import get_train_loader
    
def train_resnet(args):
    print('start training..')

    if args.data_root.endswith('/datasets'):
        data_dir_path = os.path.join(args.data_root, args.train_data)
    else:
        data_dir_path = args.data_root
    data_loader = get_train_loader(data_dir_path, batch_size=args.train_bs)
    
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    # model = torchvision.models.resnet50(pretrained=True)
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    EPOCHS = 100
    for epoch in range(1, EPOCHS+1):
        losses = []
        running_loss = 0
        for i, inp in enumerate(data_loader):
            inputs, img_name, labels, bbox = inp

            if i == len(data_loader) - 1:
                break

            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()
        
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i%100 == 0 and i > 0:
                print(f'Loss [{epoch}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0
            
        if epoch == EPOCHS:
            torch.save(model, args.backbone_weight)


        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
                
    print('Training Done')


if __name__ == '__main__':
    args = get_args(work_path)
    train_resnet(args)