from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
import pandas as pd

import time
import os
import copy
import csv

cudnn.benchmark = True
##########################################################---IMPORTS---############################################################################

load_presaved_model = True

dataset = 'FairFace_AWIB_Equal'

data_dir = 'Z:\RR\Final\Datasets\FairFace_AWIB_Equal'
fold_dir = 'Z:\RR\Final\Datasets\FairFace_AWIB_Equal\FairFace_AWIB_Equal'
# data_dir = 'Z:\RR\Final\Datasets\FairFace_All_Races'

fold = 1
#v8 and onwards is the FairFace testing batch
model_name = F'ResNet34_v15_{fold}'#Alpha: 1e_3, Step_size: 100, before we decrease alpha

model_save_path = os.path.join(os.path.join('~/Models',dataset),model_name + '.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_acc = 0.0
	
pre_processing = 'Resized_RandomHorizontalFlip'

k_folds = 5
batch_size = 128
num_workers = 8
optimizer = "Adam"

lr = 0.001
momentum= 0.9

step_size=15
gamma=0.05

max_epochs = 200
patience = 10
min_delta = 0

image_size = 224

##########################################################---Hyper-parameters---############################################################################

# Data augmentation and normalization for training
# Just normalization for validation
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transforms = {
	'train': transforms.Compose([ #compose several transforms together
		# transforms.RandomResizedCrop(224),
		transforms.Resize(image_size),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
		# transforms.Normalize(mean, std)
	]),
	'val': transforms.Compose([
		transforms.Resize(image_size),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std) # normalize an image with mean and std
	]),
}

##########################################################---Glabal Variables---######################################################################
class EarlyStopping():
	def __init__(self):

		self.patience = patience
		self.min_delta = min_delta
		self.counter = 0
		self.early_stop = False
		self.val_loss = 0
		self.train_loss = 0
		self.val_losses = []
  
	def set_train_loss(self,t_loss):
		self.train_loss = t_loss
  
	def set_val_loss(self,v_loss):
		self.val_loss = v_loss
		self.val_losses.append(v_loss)

	def check_early_stop(self):
		if (self.val_loss - self.train_loss) > self.min_delta:
			if len(self.val_losses) > 1 and self.val_losses[-1] > self.val_losses[-2]:

				self.counter +=1
				print(F'Encountered an early stopping criteria \nTrain Loss:{self.train_loss}\nVal Loss:{self.val_loss}\n')
			if self.counter >= self.patience:  
				self.early_stop = True
		else:
			self.counter = 0
			

def save_checkpoint(model, optimizer, save_path, epoch):
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch
	}, save_path)
	
def load_checkpoint(model, optimizer, load_path):
	if torch.cuda.is_available():
		
		checkpoint = torch.load(load_path)
	else:
		checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
		
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']

	return model, optimizer, epoch

def write_hyperparameters():
	path = F'~/Results/{dataset}/hyperparameters_{model_name}.csv'
 
	with open(os.path.expanduser(path),'a') as csvfile:
		fieldnames = ['training_dataset','Optimizer','image_size','k_folds','fold','pre_processing','batch_size','num_workers','lr','momentum','step_size','gamma','max_epochs','patience','min_delta']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow({'training_dataset':dataset,'Optimizer':optimizer,'image_size':image_size,'k_folds':k_folds,'fold':fold,'pre_processing':pre_processing,'batch_size':batch_size,'num_workers':num_workers,
				   'lr':lr,'momentum':momentum,
				   'step_size':step_size,'gamma':gamma,
				   'max_epochs':max_epochs,'patience':patience,'min_delta':min_delta})

def write_step_results(training_loss,training_acc,validation_loss,validation_acc,fold):

        path = F'~/Results/{dataset}/results_{model_name}.csv'

        with open(os.path.expanduser(path),'a') as csvfile:
                fieldnames = ['fold','epoch', 'train_loss','train_acc','val_loss','val_acc']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                for i in range(len(training_loss)):
                        writer.writerow({'fold':fold,'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i].item(),'val_loss':validation_loss[i],'val_acc':validation_acc[i].item()})

def write_results(time_elapsed,training_loss,training_acc,validation_loss,validation_acc,fold):
	
	path = F'~/Results/{dataset}/results_{model_name}.csv'
 
	with open(os.path.expanduser(path),'a') as csvfile:
		fieldnames = ['fold','epoch', 'train_loss','train_acc','val_loss','val_acc']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		for i in range(len(training_loss)):
			writer.writerow({'fold':fold,'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i].item(),'val_loss':validation_loss[i],'val_acc':validation_acc[i].item()})
   
	path = F'~/Results/{dataset}/time_results_{model_name}.csv'

 
	with open(os.path.expanduser(path),'a') as csvfile:
		fieldnames = ['fold','model', 'time_taken']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		writer.writerow({'fold':fold,'model':model_name,'time_taken':time_elapsed})

def reset_weights(m):
	'''
	Resetting model weights to avoid
	weight leakage.
	'''
	for layer in m.children():
		
		if hasattr(layer, 'reset_parameters'):
			# print(f'Reset trainable parameters of layer = {layer}')
			layer.reset_parameters()
   
def train(model, criterion, optimizer, scheduler,train_loader,val_loader,dataset_size_train,dataset_size_val, num_epochs,best_acc):
	since = time.time()
	early_stopping = EarlyStopping()

	best_model_wts = copy.deepcopy(model.state_dict())
 
	training_loss = []
	training_acc = []
 
	validation_loss = []
	validation_acc = []


	for epoch in range(0,num_epochs):
		print(f'Epoch {epoch}/{num_epochs - 1}')
		print('-' * 10)

		# Each epoch has a training and validation phase
		for phase in ['train', 'val']:
			if phase == 'train':
				model.train()  # Set model to training mode
			else:
				model.eval()   # Set model to evaluate mode

			running_loss = 0.0
			running_corrects = 0

			# Iterate over data.
			if phase == 'train':
				data_ldr = train_loader
			else:
				data_ldr = val_loader
				
			for inputs, labels in data_ldr:
				inputs = inputs.to(device)
				labels = labels.to(device)

				# zero the parameter gradients
				optimizer.zero_grad()

				# forward
				# track history if only in train
				with torch.set_grad_enabled(phase == 'train'):
					outputs = model(inputs)
					_, preds = torch.max(outputs, 1)
					loss = criterion(outputs, labels)

					# backward + optimize only if in training phase
					if phase == 'train':
						loss.backward()
						optimizer.step()

				# statistics
				running_loss += loss.item() * inputs.size(0)
				running_corrects += torch.sum(preds == labels.data)
			if phase == 'train':
				scheduler.step()
				epoch_loss = running_loss / dataset_size_train
				epoch_acc = running_corrects.double() / dataset_size_train
			else:
				epoch_loss = running_loss / dataset_size_val
				epoch_acc = running_corrects.double() / dataset_size_val

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
			print('',flush=True,end = '')
   
   
			if phase == 'train':
				training_loss.append(epoch_loss)
				training_acc.append(epoch_acc)
				early_stopping.set_train_loss(epoch_loss)
	
			else:
				validation_loss.append(epoch_loss)
				validation_acc.append(epoch_acc)
				write_results(training_loss,training_acc,validation_loss,validation_acc,fold)
	
			save_checkpoint(model, optimizer, os.path.expanduser(model_save_path),epoch)
		
			if phase == 'val':
				# deep copy the model
				if epoch_acc > best_acc:
					best_acc = epoch_acc
					best_model_wts = copy.deepcopy(model.state_dict())
				
				early_stopping.set_val_loss(epoch_loss)
				early_stopping.check_early_stop()
		
		if early_stopping.early_stop:
			print("Early stopping")
			break

		print()

	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')

	# load best model weights
	model.load_state_dict(best_model_wts)
	save_checkpoint(model, optimizer, os.path.expanduser(model_save_path),epoch)
 
	return model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc

##########################################################---Function To Train Model---######################################################################
def get_modal(class_names):
	weights = models.ResNet34_Weights.DEFAULT
	model = models.resnet34(weights = weights)
	num_ftrs = model.fc.in_features

	model.fc = nn.Linear(num_ftrs, len(class_names))
 
	return model

def get_indices():
	path = os.path.join(fold_dir,F"train_idx_{fold}.csv")
	path = os.path.expanduser(path)
	train_idx = np.genfromtxt(path,delimiter=',').astype(int)
	
	path = os.path.join(fold_dir,F"val_idx_{fold}.csv")
	path = os.path.expanduser(path)
	val_idx = np.genfromtxt(path,delimiter=',').astype(int)
	
	return train_idx,val_idx


def k_fold( class_names,image_datasets,train_idx,val_idx, max_epochs,best_acc):
	  # K-fold Cross Validation model evaluation

	dataset= torch.utils.data.ConcatDataset([image_datasets['train']])

	print('------------fold no---------{}----------------------'.format(fold))


	train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
	val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
 
	print(F"Length of training indices {len(train_idx)}")
	print(F"Length of val indices {len(val_idx)}")


	train_loader = torch.utils.data.DataLoader(
						dataset, 
						batch_size=batch_size,num_workers=num_workers, sampler=train_subsampler)
	val_loader = torch.utils.data.DataLoader(
						dataset,
						batch_size=batch_size,num_workers=num_workers, sampler=val_subsampler)
	dataset_size_train = len(train_loader) * batch_size
	dataset_size_val = len(val_loader) * batch_size


	model = get_modal(class_names)
	optimizer = optim.Adam(model.parameters(), lr=lr)
	exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma= gamma)
	criterion = nn.CrossEntropyLoss()
	model = model.to(device)
  
  
	trained_model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc = train(model, criterion, optimizer, exp_lr_scheduler,
																				   train_loader,val_loader,dataset_size_train,dataset_size_val,
																				   max_epochs,best_acc)

	write_results(time_elapsed,training_loss,training_acc,validation_loss,validation_acc,fold)
  
def init_training(best_acc):
	print('Setting up Training')
	path = F'~/Results/{dataset}/results_{model_name}.csv'
	open(os.path.expanduser(path),'w+')
	path = F'~/Results/{dataset}/time_results_{model_name}.csv'
	open(os.path.expanduser(path),'w+')
	path = F'~/Results/{dataset}/hyperparameters_{model_name}.csv'
	open(os.path.expanduser(path),'w+')
	print('Done checking paths')
	# Ensuring the paths for saving modal progress are working
	write_hyperparameters()

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
											data_transforms[x])
					for x in ['train']}

	class_names = image_datasets['train'].classes

	
	print('Training Started \n\n')
 
	train_idx,val_idx = get_indices()
	k_fold(class_names,image_datasets,train_idx,val_idx,max_epochs,best_acc)
 
   
# if __name__ == '__main__':
	
print('',flush=True,end = '')
init_training(best_acc)

	##########################################################---Initializing Model---######################################################################
