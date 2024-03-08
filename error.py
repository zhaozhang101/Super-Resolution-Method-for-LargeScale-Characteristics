import sys
import time
import numpy as np
import pandas as pd
import torch.optim as optim
import os
import torch
from pathlib import Path
import mlt_dataprocess
import mlt_loss
import mlt_model

def error(model, args, train_loader, test_loader):
    # define
    work_dir = os.path.dirname(os.path.abspath(__file__))
    Result = os.path.join(work_dir,'result')
    train_loss = np.repeat(np.nan, 6, axis=0).tolist()
    test_loss = np.repeat(np.nan, 7, axis=0).tolist()
    test_std_error = np.repeat(np.nan, 5, axis=0).tolist()
    test_rmse = np.repeat(np.nan, 2, axis=0).tolist()
    test_ame = np.repeat(np.nan, 2, axis=0).tolist()
    train_std_error = np.repeat(np.nan, 5, axis=0).tolist()
    train_lossfunction1 = mlt_loss.traloss0()
    train_lossfunction2 = mlt_loss.traloss2()
    test_lossfunction1 = mlt_loss.tesloss0()
    test_lossfunction2 = mlt_loss.tesloss2()
    test_rmsefunction = mlt_loss.tesloss_rmse()
    test_amefunction = mlt_loss.tesloss_ame()

    work_dir = os.path.dirname(os.path.abspath(__file__))
    Result = os.path.join(work_dir,'result')
    train_loss = np.repeat(np.nan, 2, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    test_loss = np.repeat(np.nan, 2, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    test_std_error = np.repeat(np.nan, 2, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    train_std_error = np.repeat(np.nan, 2, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    train_lossfunction1 = mlt_loss.traloss0()
    train_lossfunction2 = mlt_loss.traloss2()
    test_lossfunction1 = mlt_loss.tesloss0()
    test_lossfunction2 = mlt_loss.tesloss2()
    stdfunction = mlt_loss.Std()
    Uncertainty = mlt_loss.bploss(len(args.target_index), args).to(args.device)

    # scheduler
    # scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    # training
    since = time.time()
    sys.stdout = mlt_dataprocess.Record(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.logname))
    csvfile = Path(os.path.join('result', args.csvname))
    if csvfile.is_file():
        os.remove(csvfile)

    # 顺序是 K phi theta p t los
    list = ['epoch', 'P', 'P_std']   # 想要不同的结果这里需要改！！！！！！！！！！！！
    data = pd.DataFrame([list])
    data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

    print("*************************第一阶段:********************************* ")

    work_dir = os.path.dirname(os.path.abspath(__file__))
    Data = os.path.join(work_dir,'result')

    characteristic_index = np.array([0, 1, 2, 3, 4, 5, 6])

    model = mlt_model.mltask(in_channels=len(characteristic_index), out_channels=64)
    model = mlt_dataprocess.resume(model, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', 'PhiTheta_scale=8.pth'))
    model = model.to('cuda')
    for epoch in range(args.epochs_P1):
        P1_epoch_trainloss = torch.Tensor([0]).to(args.device)
        P1_epoch_teststderror = torch.Tensor([0]).to(args.device)
        P1_epoch_testloss = torch.Tensor([0]).to(args.device)
        P1_epoch_testrmse = torch.Tensor([0]).to(args.device)
        P1_epoch_testame = torch.Tensor([0]).to(args.device)
        # optimizer
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': Uncertainty.parameters(), 'lr': args.lr}])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)
                power = model(data)   # 想要不同的结果这里需要改！！！！！！！！！！！！
                
                loss3power = test_lossfunction1(power, label[:, 1:3, :, :], mask)
                
                loss = loss3power
                
                std3power = stdfunction(power, label[:, 1:3, :, :], mask)
                
                std = std3power

                ame3power = test_amefunction(power, label[:, 1:3, :, :], mask)

                ame = ame3power

                rmse3power = test_rmsefunction(power, label[:, 1:3, :, :], mask)

                rmse = rmse3power

                P1_epoch_testloss = P1_epoch_testloss + loss / len(test_loader)
                P1_epoch_teststderror = P1_epoch_teststderror + std / len(test_loader)
                P1_epoch_testame = P1_epoch_testame + ame / len(test_loader)
                P1_epoch_testrmse = P1_epoch_testrmse + rmse / len(test_loader)
                
        print(P1_epoch_teststderror)
        # result recording
        counter = 0
        
        
        for index_I in range(5):
            if len([k for k in range(len(args.list_index)) if args.list_index[k] == index_I + 1]) > 0:
                test_loss[counter] = P1_epoch_testloss[counter].item()
                test_std_error[counter] = P1_epoch_teststderror[counter].item()
                test_ame[counter] = P1_epoch_testame[counter].item()
                test_rmse[counter] = P1_epoch_testrmse[counter].item()
                counter = counter + 1
        

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [180, 180]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        # train_loss = (np.array(train_loss) * [100]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        test_std_error = (np.array(test_std_error) * [180, 180]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        test_ame = (np.array(test_ame) * [180, 180]).tolist()
        test_rmse = (np.array(test_rmse) * [180, 180]).tolist()
        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

        
        print(
            f"Epoch : {epoch + 1} (testloss)  phi :{test_loss[0]:.4f}   theta :{test_loss[1]:.4f}")   # 想要不同的结果这里需要改！！！！！！！！！！！！
        print(
            f"Epoch : {epoch + 1} (test_std_error)  phi :{test_std_error[0]:.4f}   theta :{test_std_error[1]:.4f}")   # 想要不同的结果这里需要改！！！！！！！！！！！！
        print(
            f"Epoch : {epoch + 1} (test_ame)  phi :{test_ame[0]:.4f}   theta :{test_ame[1]:.4f}")   # 想要不同的结果这里需要改！！！！！！！！！！！！
        print(
            f"Epoch : {epoch + 1} (test_rmse)  phi :{test_rmse[0]:.4f}   theta :{test_rmse[1]:.4f}\n")   # 想要不同的结果这里需要改！！！！！！！！！！！！