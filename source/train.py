import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.autograd import Variable
from collections import OrderedDict

DEVICE = 'cuda:0'

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
    metrics = []
    for sample in data:
        image = Variable(sample['image'].float().to(DEVICE))
        label = Variable(sample['label'].float().squeeze().to(DEVICE))
        output = model(image)
        batch_size = label.size(0)
        loss = loss_fn(output, label)
        losses.append(loss.data.item())
        metric = metric_fn(output.cpu().data.numpy(), label.cpu().data.numpy())
        metrics.append(metric)
    del image, label, output
    torch.cuda.empty_cache()
    mean_loss = round(np.mean(losses),5)
    mean_metric = round(np.mean(metrics),5)
    model.train()
    return mean_loss, mean_metric

def trainModel(model, train_data, valid_data, loss_fn, metric_fn, optimizer, save_path, epochs, batch, scheduler):
    checkpoint = np.inf
    logger = save_path.replace('model.pt','log.txt')
    logger = open(logger,'w')
    logger.write('epoch,train_loss,train_metric,valid_loss,valid_metric\n')
    for epoch in range(epochs):
        model.train()
        tq = tqdm(total=len(train_data) * batch, ncols=0, disable=False)
        losses = []
        metrics = []
        for sample in train_data:
            image = Variable(sample['image'].float().to(DEVICE))
            label = Variable(sample['label'].float().squeeze().to(DEVICE))
            output = model(image)
            batch_size = label.size(0)
            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            losses.append(loss.data.item())
            train_loss = round(np.mean(losses),5)
            tq.update(batch_size)
            tq.set_postfix(trn_ls='{:.5f}'.format(train_loss))
            metric = metric_fn(output.cpu().data.numpy(), label.cpu().data.numpy())
            metrics.append(metric)
        del image, label, output
        torch.cuda.empty_cache()
        train_metric = round(np.mean(metrics),5)
        valid_loss, valid_metric = validModel(model, valid_data, loss_fn, metric_fn)
        postfix = OrderedDict()
        postfix['trn_ls'] = '{:.5f}'.format(train_loss)
        postfix['trn_mt'] = '{:.5f}'.format(train_metric)
        postfix['val_ls'] = '{:.5f}'.format(valid_loss)
        postfix['val_mt'] = '{:.5f}'.format(valid_metric)
        tq.set_postfix(**postfix)
        tq.close()
        logger.write(f'{epoch},{train_loss},{train_metric},{valid_loss},{valid_metric}\n')
        scheduler.step(valid_loss)
        if valid_loss < checkpoint:
            checkpoint = valid_loss
            saveModel(epoch, model, valid_loss, save_path)
    logger.close()
    return None

