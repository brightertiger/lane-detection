import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

def trainModel(model, train_data, valid_data, loss_fn, metric_fn, optimizer, scheduler, save_path, epochs, batch=6):
    best_loss = float('inf')
    
    log_path = save_path.replace('model.pt', 'log.txt')
    with open(log_path, 'w') as f:
        f.write('epoch,train_loss,train_metric,valid_loss,valid_metric\n')
    
    for epoch in range(epochs):
        model.train()
        train_losses, train_metrics = [], []
        
        pbar = tqdm(total=len(train_data) * batch)
        for sample in train_data:
            image = sample['image'].float().cuda()
            label = sample['label'].float().cuda()
            
            output = model(image)
            loss = loss_fn(output, label)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_metrics.append(metric_fn(output, label))
            
            pbar.update(batch)
            pbar.set_postfix({
                'trn_ls': f'{np.mean(train_losses):.5f}',
                'trn_mt': f'{np.mean(train_metrics):.5f}'
            })
        
        model.eval()
        valid_losses, valid_metrics = [], []
        
        with torch.no_grad():
            for sample in valid_data:
                image = sample['image'].float().cuda()
                label = sample['label'].float().cuda()
                
                output = model(image)
                loss = loss_fn(output, label)
                
                valid_losses.append(loss.item())
                valid_metrics.append(metric_fn(output, label))
        
        train_loss = np.mean(train_losses)
        train_metric = np.mean(train_metrics)
        valid_loss = np.mean(valid_losses)
        valid_metric = np.mean(valid_metrics)
        
        pbar.set_postfix({
            'trn_ls': f'{train_loss:.5f}',
            'trn_mt': f'{train_metric:.5f}',
            'val_ls': f'{valid_loss:.5f}',
            'val_mt': f'{valid_metric:.5f}'
        })
        pbar.close()
        
        with open(log_path, 'a') as f:
            f.write(f'{epoch+1},{train_loss:.6f},{train_metric:.6f},{valid_loss:.6f},{valid_metric:.6f}\n')
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': valid_loss,
                'metric': valid_metric
            }, save_path)
        
        scheduler.step(valid_loss)

