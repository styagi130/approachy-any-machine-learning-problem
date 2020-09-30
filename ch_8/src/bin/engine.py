import torch
import tqdm

def train_epoch(data_loader, model, optimizer, criterion, device, fold, epoch):
	"""
		Function to run one training epoch
		:param data_loader: Torch.utils.data.DataLoader class
		:param model: model to train
		:param optimizer: optimizer, i.e ADAM, SGD. This is also torch class
		:param criterion: Loss function
		:param device: Device to train model on
		:param fold: Fold number
		:param epoch: epoch number
	"""
	model.train()
	for inputs, input_lens, labels in tqdm.tqdm(data_loader, ncols=100, desc=f"train-- F: {fold} -- E: {epoch}"):
		inputs = inputs.to(device)
		labels = labels.to(device)
		#input_lens = input_lens.to(device)

		optimizer.zero_grad()
		preds = model(inputs, input_lens)
		
		loss = criterion(preds, labels.unsqueeze(1))
		loss.backward()
		optimizer.step()

@torch.no_grad()
def eval_step(data_laoder, model, device, fold):
	"""
		Function to run evaluation step
		:param data_loader: torch.utils.data.DataLoader class
		:param model: Model to run eval on
		:param device: Device used for training 
	"""
	model.eval()
	true_labels, pred_labels = [], []
	for inputs, input_lens, labels in tqdm.tqdm(data_laoder, ncols=100, desc=f"eval--- F: {fold} -- E: n"):
		inputs = inputs.to(device)
		labels = labels.to(device)
		#input_lens = input_lens.to(device)

		preds = model(inputs, input_lens)

		true_labels.extend(labels.detach().squeeze().cpu().numpy().tolist())
		pred_labels.extend(preds.detach().squeeze().cpu().numpy().tolist())

	return true_labels, pred_labels
