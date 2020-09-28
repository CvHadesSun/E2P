import model
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from loss import loss_l2 as Loss
import cv2
import dataset
import os
# to train refinenet 





###############
def train(model,train_loader,optimizer):
    #train
    # print("training...")
    model.train()
    epoch_loss=0
    # i=0
    for input,label in train_loader:
        # i+=1
        # print(i)

        input=input.cuda()
        # print(input.shape)
        label=label.cuda()

        output=model(input)
        loss=Loss(output,label)
        epoch_loss+=loss

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

 
    return model,epoch_loss/len(train_loader)




def test(model,val_loader):
    val_loss=0
    model.eval()
    with torch.no_grad():
        # print(len(val_loader))
        # i=0
        for input,label in val_loader: 
            # i+=1
            # print(i)
            input=input.cuda()
            label=label.cuda()
            output=model(input)

            # print(output)
            loss=Loss(output,label)
            val_loss+=loss
        

        return val_loss/len(val_loader)


def main():

    #Hyper parameters

    initial_lr = 0.01
    momentum = 0.9
    weight_decay = 5e-4
    num_epoch = 5000
    use_gpu = True
    gen_kp_gt = False
    number_point = 8
    modulating_factor = 1.0
    in_channels=8
    num_keypoints=8
    batch_size=32
    train_pred_file='./../tools/results/ape_101/refinement/train_8_101_pred.json'
    train_gt_file='./../datasets/LINEMOD/ape/ape_val_gen.json'
    val_pred_file='./../tools/results/ape_101/refinement/val_8_101_pred.json'
    val_gt_file='./../datasets/LINEMOD/ape/ape_val_gen.json'

    model_savepath='./../tools/results/ape_101/refinement/model_regression'

    modelpath='./../tools/results/ape_101/refinement/model_regression/4999.pth'
    #

    print("train data loading...")
    train_data = dataset.CoordDataset(train_pred_file, train_gt_file)
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)

    #
    print("test data loading...")
    val_data=dataset.CoordDataset(val_pred_file, val_gt_file)
    val_loader = DataLoader(val_data, shuffle=True, batch_size=batch_size)
                
    #
    #     
    # #build RefineNet
    print("RefineNet building...")
    Feature_extractor=model.KeypointRCNNFeatureExtractor(in_channels)
    Predictor=model.KeypointRCNNPredictor2(16,num_keypoints)
    RefineNet=model.RefineNet(Feature_extractor,Predictor)
    print("model build done ")     

    RefineNet.load_state_dict(torch.load(modelpath))
    #
    optimizer = torch.optim.SGD(RefineNet.parameters(), lr=initial_lr, 
                                momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                [int(0.5*num_epoch), int(0.75*num_epoch),
                                    int(0.9*num_epoch)], gamma=0.1)
    #gpu 
    RefineNet.cuda()
            

    # val_loss=test(RefineNet,val_loader)
    print('training...')
    for epoch in range(num_epoch):
        #train
        scheduler.step()
        RefineNet,trian_loss=train(RefineNet,train_loader,optimizer)

        # print(len(train_loader))
        print("epoch:{}".format(epoch)+"="*50+"train loss:{}".format(trian_loss))
        #test

        val_loss=test(RefineNet,val_loader)
        # print(val_loader)
        print("epoch:{}".format(epoch)+"="*50+"validate loss:{}".format(val_loss))


        #save model 
        modelname=str(epoch)+'.pth'
        modelpath=os.path.join(model_savepath,modelname)
        torch.save(RefineNet.state_dict(), modelpath)

    print("train done")

        
    return True



# 
if __name__ == "__main__":
    main()
    