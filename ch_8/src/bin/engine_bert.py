import torch
import tqdm

def train_step(data_loader, model, optimizer, scheduler, criterion, device, epoch):
    """
        Function to run bert training step
        :param data_loader: Torch.utils.data.DataLoader class
        :param model: model to train
        :param optimizer: optimizer, i.e ADAM, SGD. This is also torch class
        :param scheduler: scheduler to update learning rate
        :param criterion: Loss function
        :param device: Device to train model on
        :param epoch: Track the epoch number
    """
    model.train()
    for ids, ids_masks, ids_types, labels in tqdm.tqdm(data_loader,total = len(data_loader), ncols=50, desc=f"training_epoch: {epoch}"):
        ids = ids.to(device)
        ids_masks = ids_masks.to(device)
        ids_types = ids_types.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        
        preds = model(ids, ids_masks, ids_types)
        
        loss = criterion(preds, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        scheduler.step()


@torch.no_grad()
def eval_step(data_loader, model, device, epoch):
    """
        Function to run evaluation step
        :param data_loader: Torch.utils.data.DataLoader class
        :param model: model to train
        :param device: Device to train model on
        :param epoch: Track the epoch number
    """
    model.eval()
    true_labels, pred_labels = [],[]
    for ids, ids_masks, ids_types, labels in tqdm.tqdm(data_loader, total = len(data_loader), ncols=50, desc=f"val_epoch: {epoch}"):
        ids = ids.to(device)
        ids_masks = ids_masks.to(device)
        ids_types = ids_types.to(device)
        labels = labels.to(device)

        preds = model(ids, ids_masks, ids_types)

        true_labels.extend(labels.cpu().detach().numpy().tolist())
        pred_labels.extend(preds.squeeze(1).cpu().detach().numpy().tolist())

    return true_labels, pred_labels
