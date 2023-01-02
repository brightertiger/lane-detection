import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict

DEVICE = 'cuda:0'

def reduceLoss(loss):
    return loss.sum() / loss.shape[0]

def saveModel(epoch, model, loss, path):
    results = {}
    results['epoch'] = epoch
    results['model_state_dict'] = model.state_dict()
    results['loss'] = loss
    torch.save(results, path)
    return None

def validModel(model, data, loss_fn, metric_fn):
    model.eval()
    losses = []
    label_array = []
    pred_array = []
    for sample in data:
        image = Variable(sample['image'].float().to(DEVICE))
        label = Variable(sample['label'].float().squeeze().to(DEVICE))
        regress, classify = model(image)
        batch_size = label.size(0)
        loss = reduceLoss(loss_fn(regress, classify, label, weight))
        label_array.append(label.cpu().data.numpy())
        pred_array.append(regress.cpu().data.numpy())
        losses.append(loss.data.item())
    del image, label, regress, classify
    torch.cuda.empty_cache()
    label_array = np.hstack(label_array)
    pred_array = np.hstack(pred_array)
    metric = metric_fn(label_array, pred_array)
    mean_loss = round(np.mean(losses),5)
    metric = round(metric,5)
    model.train()
    return mean_loss, metric

def trainModel(model, train_data, valid_data, loss_fn, metric_fn, optimizer, save_path, epochs, batch, scheduler):
    checkpoint = np.inf
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        label_array = []
        pred_array = []
        for sample in train_data:
            image = Variable(sample['image'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            weight = Variable(sample['weight'].float().squeeze().to(DEVICE))
            output = model(image)
            batch_size = label.size(0)
            loss = reduceLoss(loss_fn(output, label))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),5)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            label_array.append(label.cpu().data.numpy())
            pred_array.append(regress.cpu().data.numpy())
        del image, label, regress, classify
        torch.cuda.empty_cache()
        label_array = np.hstack(label_array)
        pred_array = np.hstack(pred_array)
        train_metric = metric_fn(label_array, pred_array)
        valid_loss, valid_metric = validModel(model, valid_data, loss_fn)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.5f}'.format(train_loss)
        postfix['trn_mt'] = '{:.5f}'.format(train_metric)
        postfix['val_ls'] = '{:.5f}'.format(valid_loss)
        postfix['val_mt'] = '{:.5f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        scheduler.step(valid_loss)
        if valid_loss < checkpoint:
            checkpoint = valid_loss
            saveModel(epoch, model, valid_loss, save_path)
    return None

