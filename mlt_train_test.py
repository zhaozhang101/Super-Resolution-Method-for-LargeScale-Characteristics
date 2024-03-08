import os
import sys
import time
from pathlib import Path
from symbol import power
import mlt_dataprocess
import mlt_loss
import numpy as np
import pandas as pd
import torch
import torch.optim as optim


def train(model, args, train_loader, test_loader):
    # define
    work_dir = os.path.dirname(os.path.abspath(__file__))
    Result = os.path.join(work_dir,'result')
    train_loss = np.repeat(np.nan, 7, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    test_loss = np.repeat(np.nan, 7, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
    test_std_error = np.repeat(np.nan,5, axis=0).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！

    train_lossfunction1 = mlt_loss.traloss0()
    train_lossfunction2 = mlt_loss.traloss2()
    test_lossfunction1 = mlt_loss.tesloss0()
    test_lossfunction2 = mlt_loss.tesloss2()
    stdfunction = mlt_loss.Std()
    Uncertainty = mlt_loss.bploss(len(args.target_index), args).to(args.device)

    # training
    since = time.time()
    sys.stdout = mlt_dataprocess.Record(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.logname))
    csvfile = Path(os.path.join('result', args.csvname))
    if csvfile.is_file():
        os.remove(csvfile)

    # 顺序是 K phi theta p t los
    list = ['epoch', 'poweratio','phi','theta','P','T','TPR','FPR','K_std','phi_std','theta_std','P_std','T_std']   # 想要不同的结果这里需要改！！！！！！！！！！！！
    data = pd.DataFrame([list])
    data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

    print("*************************第一阶段:********************************* ")
    for epoch in range(args.epochs_P1):
        P1_epoch_trainloss = torch.Tensor([0]).to(args.device)
        P1_epoch_teststderror = torch.Tensor([0]).to(args.device)
        P1_epoch_testloss = torch.Tensor([0]).to(args.device)
        # optimizer
        optimizer = optim.Adam([
            {'params': model.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay},
            {'params': Uncertainty.parameters(), 'lr': args.lr}])

        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)
            label = Label.to(args.device)

            mask = mask.to(args.device)
            thephi, poweratio, power, delay, los = model(data, args)   # 想要不同的结果这里需要改！！！！！！！！！！！！
            loss1poweratio = train_lossfunction1(poweratio, label[:, 0:1, :, :], mask)  # c
            loss2thephi = train_lossfunction1(thephi, label[:, 1:3, :, :], mask)
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)  # 1

            
            """std error plus"""
            std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
            std2thephi = stdfunction(thephi, label[:, 1:3, :, :], mask)
            std3power = stdfunction(power, label[:, 3:4, :, :], mask)
            std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)

            loss = torch.cat((loss1poweratio,loss2thephi, loss3power, loss4delay, lossLOS), dim=0)  # 想要不同的结果这里需要改！！！！！！！！！！！！
            Std = torch.cat((std1poweratio,std2thephi, std3power, std4delay), dim=0)  # 想要不同的结果这里需要改！！！！！！！！！！！！

            bploss = Uncertainty(loss, Std)
            optimizer.zero_grad()
            bploss.backward()
            optimizer.step()
            P1_epoch_trainloss = P1_epoch_trainloss + loss / len(train_loader)

        # print(model.state_dict()['net1.Convv.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)
                thephi, poweratio, power, delay, los = model(data, args)   # 想要不同的结果这里需要改！！！！！！！！！！！！
                loss1poweratio = test_lossfunction1(poweratio, label[:, 0:1, :, :], mask)  # c
                loss2thephi = test_lossfunction1(thephi, label[:, 1:3, :, :], mask)
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)

                loss = torch.cat((loss1poweratio,loss2thephi,loss3power,loss4delay), dim=0)   # 想要不同的结果这里需要改！！！！！！！！！！！！

                std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
                std2thephi = stdfunction(thephi, label[:, 1:3, :, :], mask)
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)

                Std = torch.cat((std1poweratio,std2thephi,std3power,std4delay), dim=0)  # 想要不同的结果这里需要改！！！！！！！！！！！！

                P1_epoch_testloss = P1_epoch_testloss + loss / len(test_loader)
                P1_epoch_teststderror = P1_epoch_teststderror + Std / len(test_loader)
                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)
        print(P1_epoch_teststderror)
        # result recording
        counter = 0
        
        
        for index_I in range(5):
            if len([k for k in range(len(args.list_index)) if args.list_index[k] == index_I + 1]) > 0:
                train_loss[counter] = P1_epoch_trainloss[counter].item()
                test_loss[counter] = P1_epoch_testloss[counter].item()
                test_std_error[counter] = P1_epoch_teststderror[counter].item()
                counter = counter + 1

        test_loss[counter] = TPRmean.item()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        test_loss[counter+1] = FPRmean.item()   # 想要不同的结果这里需要改！！！！！！！！！！！！

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100,180,180,100,100,100,100]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        test_std_error = (np.array(test_std_error) * [100,180,180,100,100]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！
        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)


        print(
            f"Epoch : {epoch + 1} (trainloss)  p :{train_loss[0]:.4f}    ")   # 想要不同的结果这里需要改！！！！！！！！！！！！
        print(
            f"Epoch : {epoch + 1} (testloss)  P :{test_loss[0]:.4f}    ")   # 想要不同的结果这里需要改！！！！！！！！！！！！
        print(
            f"Epoch : {epoch + 1} (test_std_error)  p_std :{test_std_error[0]:.4f}   \n")   # 想要不同的结果这里需要改！！！！！！！！！！！！

    print("*************************第二阶段:********************************* ")
    args.period = 2
    for epoch in range(args.epochs_P2):
        P2_epoch_trainloss = torch.Tensor([0]).to(args.device)
        P2_epoch_teststderror = torch.Tensor([0]).to(args.device)
        P2_epoch_testloss = torch.Tensor([0]).to(args.device)

        # optimizer
        optimizer = optim.Adam([
            {'params': model.net2.parameters(), 'lr': args.lr, 'weight_decay': args.weight_decay}])
        for Data, Label, mask in (train_loader):
            data = Data.to(args.device)
            label = Label.to(args.device)
            mask = mask.to(args.device)
            thephi, poweratio, power, delay, los = model(data, args)   # 想要不同的结果这里需要改！！！！！！！！！！！！
            loss1poweratio = train_lossfunction1(poweratio, label[:, 0:1, :, :], mask)  # c
            loss2thephi = train_lossfunction1(thephi, label[:, 1:3, :, :], mask)
            loss3power = train_lossfunction1(power, label[:, 3:4, :, :], mask)
            loss4delay = train_lossfunction1(delay, label[:, 4:5, :, :], mask)
            lossLOS = train_lossfunction2(los, label[:, [-1], :, :], mask).unsqueeze(0)  # 1

            loss = torch.cat((loss1poweratio, loss2thephi,loss3power, loss4delay, lossLOS), dim=0)   # 想要不同的结果这里需要改！！！！！！！！！！！！


            optimizer.zero_grad()

            torch.sum(loss1poweratio ).backward()
            torch.sum(loss2thephi ).backward()
            torch.sum(loss3power ).backward()   # 想要不同的结果这里需要改！！！！！！！！！！！！
            torch.sum(loss4delay ).backward()
            torch.sum(lossLOS).backward()
            optimizer.step()

            P2_epoch_trainloss = P2_epoch_trainloss + loss / len(train_loader)

        # print(model.state_dict()['net2.power.0.bias'])

        with torch.no_grad():
            TPRmean = torch.Tensor([0]).to(args.device)
            FPRmean = torch.Tensor([0]).to(args.device)
            for Data, Label, mask in test_loader:
                data = Data.to(args.device)
                label = Label.to(args.device)
                mask = mask.to(args.device)
                thephi, poweratio, power, delay, los = model(data, args)   # 想要不同的结果这里需要改！！！！！！！！！！！！
                loss1poweratio = test_lossfunction1(poweratio, label[:, 0:1, :, :], mask)  # c
                loss2thephi = test_lossfunction1(thephi, label[:, 1:3, :, :], mask)
                loss3power = test_lossfunction1(power, label[:, 3:4, :, :], mask)
                loss4delay = test_lossfunction1(delay, label[:, 4:5, :, :], mask)
                TPR, FPR = test_lossfunction2(los, label[:, [-1], :, :], mask)

                loss = torch.cat((loss1poweratio,loss2thephi,loss3power,loss4delay), dim=0)   # 想要不同的结果这里需要改！！！！！！！！！！！！

                std1poweratio = stdfunction(poweratio, label[:, 0:1, :, :], mask)
                std2thephi = stdfunction(thephi, label[:, 1:3, :, :], mask)
                std3power = stdfunction(power, label[:, 3:4, :, :], mask)
                std4delay = stdfunction(delay, label[:, 4:5, :, :], mask)

                std = torch.cat((std1poweratio, std2thephi,std3power,std4delay), dim=0)   # 想要不同的结果这里需要改！！！！！！！！！！！！

                P2_epoch_testloss = P2_epoch_testloss + loss / len(test_loader)
                P2_epoch_teststderror = P2_epoch_teststderror + std / len(test_loader)
                TPRmean += TPR / len(test_loader)
                FPRmean += FPR / len(test_loader)

        # result recording --
        counter = 0

        
        for index_I in range(5):
            if len([k for k in range(len(args.list_index)) if args.list_index[k] == index_I + 1]) > 0:
                test_loss[counter] = P2_epoch_testloss[counter].item()
                test_std_error[counter] = P2_epoch_teststderror[counter].item()
                counter = counter + 1

        test_loss[counter] = TPRmean.item()
        test_loss[counter+1] = FPRmean.item()

        time_elapsed = time.time() - since
        print("====================================================")
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        test_loss = (np.array(test_loss) * [100, 180, 180, 100, 100, 100, 100]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！

        test_std_error = (np.array(test_std_error) * [100, 180, 180, 100, 100]).tolist()   # 想要不同的结果这里需要改！！！！！！！！！！！！

        list = [epoch + 1] + test_loss
        list = list + test_std_error
        data = pd.DataFrame([list])
        data.to_csv(os.path.join(Result, args.csvname), mode='a', header=None, index=False)

        print(
            f"Epoch : {epoch + 1} (trainloss) P :{train_loss[0]:.4f}  ")
        print(
            f"Epoch : {epoch + 1} (testloss) P :{test_loss[0]:.4f}  ")
        print(
            f"Epoch : {epoch + 1} (test_std_error) P_std :{test_std_error[0]:.4f}   \n")

        # save model parameters and weights
        if epoch == args.epochs_P2 - 3:
            last_model_wts = model.state_dict()
            save_path_last = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result', args.wight_lastname)
            if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result')):
                os.mkdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'result'))
            torch.save(last_model_wts, save_path_last)
