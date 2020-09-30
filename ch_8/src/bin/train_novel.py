# Run LSTM training

import argparse
import pathlib
import gc
import tqdm
import sys

import pandas as pd
import numpy as np
import torch

from sklearn import metrics

sys.path.append("./../../")
from src.utils import io, helpers
from src.bin import engine
from src.datasets import text_classification_dataset
from src.models import lstm
from src.tokenizers import fastext_tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark=True

def main(config):
	"""
		Function to train and evaluate model
		:param config: SimpleNamespace config object
	"""
	tokenizer = fastext_tokenizer.Fasttext(config.wv_filepath, return_wv=True)

	# Load data
	train_df = pd.read_csv(config.train_fold_filepath)
	train_df.rename(columns = config.data["names"], inplace = True)
	train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

	test_df = pd.read_csv(config.test_filepath, usecols=config.data["columns"]).dropna()
	test_df.rename(columns = config.data["names"], inplace = True)
	test_feature_label_list = zip(test_df.loc[:,"feature"].values, test_df.loc[:, "labels"].values)
	test_dataset = text_classification_dataset.TextClassifier(test_feature_label_list, tokenizer)
	del test_feature_label_list, test_df

	folds = np.unique(train_df.loc[:, "fold"].values)
	running_roc_auc_avg = 0
	running_accuracy_avg = 0
	#folds = [3]
	for fold in folds:
		print (f"Starting training for fold: {fold}")
		fold_model_dir = config.model_dirpath / f"{config.model_name}/{fold}"
		if not fold_model_dir.exists():
			print (f"Creating model_dir {str(fold_model_dir)}")
			fold_model_dir.mkdir(parents=True)

		best_model_path = fold_model_dir / "best_model.ckpt"
		best_state_dict = None

		embedding_dims = tokenizer.embedding_dim
		# Get model class
		model = helpers.dispatch_model(config, embedding_dims)
		model.to(device)

		optimizer = torch.optim.RMSprop(model.parameters(), lr=config.lr)#RMSprop
		criterion = torch.nn.BCELoss()

		val_index = train_df.loc[:, "fold"] == fold
		train_index = ~val_index
		
		train_feature_label_list = zip(train_df.loc[train_index,"feature"].values, train_df.loc[train_index, "labels"].values)
		val_feature_label_list = zip(train_df.loc[val_index,"feature"].values, train_df.loc[val_index, "labels"].values)

		train_dataset = text_classification_dataset.TextClassifier(train_feature_label_list, tokenizer)
		val_dataset = text_classification_dataset.TextClassifier(val_feature_label_list, tokenizer)

		train_dataloader = torch.utils.data.DataLoader(
														train_dataset, 
														collate_fn=train_dataset.collate_fn,
														shuffle=False,
														batch_size=config.train_batch_size,
														num_workers=config.num_workers
													  )
		val_dataloader = torch.utils.data.DataLoader(
														val_dataset,
														collate_fn=val_dataset.collate_fn,
														shuffle=False,
														batch_size=config.eval_batch_size,
														num_workers=config.num_workers
													)

		best_score = 0
		early_stopping_counter = 0
		for epoch in range(config.epochs):
			engine.train_epoch(train_dataloader, model, optimizer, criterion, device, fold, epoch)
			true_labels, pred_labels = engine.eval_step(val_dataloader, model, device, fold)

			pred_labels = [1 if score >0.5 else 0 for score in pred_labels]
			accuracy_score = metrics.accuracy_score(true_labels, pred_labels)
			if accuracy_score > best_score:
				best_score = accuracy_score
				early_stopping_counter = 0
				roc_auc = metrics.roc_auc_score(true_labels, pred_labels)

				torch.save(model.state_dict(), best_model_path)
				print (f"FOLD: {fold}, epoch: {epoch}, val roc_auc score: {roc_auc}, val_accuracy is: {accuracy_score}")
			else:
				early_stopping_counter += 1
			
			if early_stopping_counter > config.early_stopping_backsteps and epoch > config.start_early_stop:
				break

		### Run TEST Results

		# Release memory
		del train_feature_label_list, val_feature_label_list
		del train_dataset, val_dataset
		del train_dataloader, val_dataloader
		del model
		del optimizer
		gc.collect()
		torch.cuda.empty_cache()

		# Load best model
		embedding_dims = tokenizer.embedding_dim
		model = helpers.dispatch_model(config, embedding_dims)
		model.to(device)
		model.load_state_dict(torch.load(best_model_path))
		model.to(device=device)

		print (f"Running tests on unseen data for fold: {fold}")
		test_dataloader = torch.utils.data.DataLoader(
														test_dataset,
														collate_fn=test_dataset.collate_fn,
														shuffle=False,
														batch_size=config.eval_batch_size,
														num_workers=config.num_workers
													)
		test_true, test_pred = engine.eval_step(test_dataloader, model, device, fold)
		roc_auc = metrics.roc_auc_score(test_true, test_pred)
		
		pred_labels = [1 if score >0.5 else 0 for score in test_pred]
		accuracy_score = metrics.accuracy_score(test_true, pred_labels)

		print (f"FOLD: {fold}, test roc_auc score: {roc_auc}, test_accuracy_score: {accuracy_score}")
		print (f"FOLD: {fold}, Classification report: ")
		print (metrics.classification_report(test_true, pred_labels))
		
		running_roc_auc_avg += roc_auc
		running_accuracy_avg += accuracy_score

		## Release some memory
		del test_dataloader
		del model
		gc.collect()
		torch.cuda.empty_cache()
		
	print (f"Average ROC AUC Score on held out test set is: {running_roc_auc_avg/len(folds)}")
	print (f"Average accuracy Score on held out test set is: {running_accuracy_avg/len(folds)}")








if __name__=="__main__":
	parser = argparse.ArgumentParser(description="Train simple LSTM model")
	parser.add_argument("config_filepath", metavar="fc", type=pathlib.Path, help="config file path")
	args = parser.parse_args()

	config = io.load_config(args.config_filepath)
	main(config)
