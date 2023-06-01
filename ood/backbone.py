import os, sys
import torch
import torchvision
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim

root_path = os.path.join(os.getcwd())
sys.path.append(root_path)
os.chdir(root_path)

from ood.args_loader import get_args
from ood.data_loader import get_train_loader
    
def train_resnet():
    print('start training..')

    args = get_args(root_path)

    data_dir_path = os.path.join(args.data_root, args.train_data)
    data_loader = get_train_loader(data_dir_path, batch_size=args.train_bs)
    
    model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
    model = model.to('cuda')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.1, patience=5)

    EPOCHS = 200
    for epoch in range(EPOCHS):
        losses = []
        running_loss = 0
        for i, inp in enumerate(data_loader):
            inputs, img_name, labels, bbox = inp

            # if len(inputs) != args.train_bs:
            #   break
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
                print(f'Loss [{epoch+1}, {i}](epoch, minibatch): ', running_loss / 100)
                running_loss = 0.0
            
        if epoch%5 == 0:
            torch.save(model, f'ood/backbone/resnet_funed_e{epoch}.pth')


        avg_loss = sum(losses)/len(losses)
        scheduler.step(avg_loss)
                
    print('Training Done')


if __name__ == '__main__':
    train_resnet()