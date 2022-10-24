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

import time
import os
import copy
import csv
#TODO make a csv file for the models hyperparameters

cudnn.benchmark = True
##########################################################---IMPORTS---############################################################################

load_presaved_model = True

dataset = 'FairFace'

data_dir = os.path.join('~/Datasets',dataset)

model_name = F'ResNet34_v1'#Alpha: 1e_3, Step_size: 100, before we decrease alpha

model_save_path = os.path.join(os.path.join('~/Models',dataset),model_name + '.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

best_acc = 0.0
	
pre_processing = 'RandomResizedCrop_RandomHorizontalFlip'

k_folds = 5

batch_size = 32
num_workers = 4

lr = 0.001
momentum= 0.9

step_size=50
gamma=0.05

max_epochs = 200
patience = 15
min_delta = 0

##########################################################---Hyper-parameters---############################################################################

# Data augmentation and normalization for training
# Just normalization for validation
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transforms = {
	'train': transforms.Compose([ #compose several transforms together
		# transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()
		# transforms.Normalize(mean, std)
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
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
  
	def set_train_loss(self,t_loss):
		self.train_loss = t_loss
  
	def set_val_loss(self,v_loss):
		self.val_loss = v_loss

	def check_early_stop(self):
		if (self.val_loss - self.train_loss) > self.min_delta:
			self.counter +=1
			print(F'Encountered an early stopping criteria \nTrain Loss:{self.train_loss}\nVal Loss:{self.val_loss}\n')
			if self.counter >= self.patience:  
				self.early_stop = True

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
	path = F'~/Results/{dataset}/hyperparameters/{model_name}.csv'
 
	with open(os.path.expanduser(path),'a') as csvfile:
		fieldnames = ['k_folds','pre_processing','batch_size','num_workers','lr','momentum','step_size','gamma','max_epochs','patience','min_delta']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerow({'pre_processing':pre_processing,'batch_size':batch_size,'num_workers':num_workers,
				   'lr':lr,'momentum':momentum,
				   'step_size':step_size,'gamma':gamma,
				   'max_epochs':max_epochs,'patience':patience,'min_delta':min_delta})

def write_results(time_elapsed,training_loss,training_acc,validation_loss,validation_acc,fold,max_epochs):
	
	path = F'~/Results/{dataset}/results_{model_name}.csv'
 
	with open(os.path.expanduser(path),'a') as csvfile:
		fieldnames = ['fold','epoch', 'train_loss','train_acc','val_loss','val_acc']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		for i in range(max_epochs):
			writer.writerow({'fold':fold,'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i].data[0],'val_loss':validation_loss[i],'val_acc':validation_acc[i].data[0]})
   
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
   
def train(model, criterion, optimizer, scheduler,train_loader,val_loader,dataset_sizes,start_epoch, num_epochs,patience):
	since = time.time()
	early_stopping = EarlyStopping()

	best_model_wts = copy.deepcopy(model.state_dict())
 
	training_loss = []
	training_acc = []
 
	validation_loss = []
	validation_acc = []


	for epoch in range(start_epoch,num_epochs):
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

			epoch_loss = running_loss / dataset_sizes[phase]
			epoch_acc = running_corrects.double() / dataset_sizes[phase]

			print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
   
			if phase == 'train':
				training_loss.append(epoch_loss)
				training_acc.append(epoch_acc)
				early_stopping.set_train_loss(epoch_loss)
	
			else:
				validation_loss.append(epoch_loss)
				validation_acc.append(epoch_acc)
	
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


def k_fold( class_names,image_datasets,start_epoch, num_epochs,k_folds):
	  # K-fold Cross Validation model evaluation
	folds = KFold(n_splits=k_folds, shuffle=True)
	for fold, (train_idx, val_idx) in enumerate(folds.split(image_datasets)):
		dataset_train = torch.utils.data.Subset(image_datasets, train_idx)
		dataset_valid = torch.utils.data.Subset(image_datasets, val_idx)
  
		train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=256, shuffle =True)
		val_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=256, shuffle = True)
		dataset_size = len(dataset_valid)

		# Sample elements randomly from a given list of ids, no replacement.
		# train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
		# val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

		# # Define data loaders for training and testing data in this fold
		# trainloader = torch.utils.data.DataLoader(
		# 					dataset, 
		# 					batch_size=256, sampler=train_subsampler)
		# testloader = torch.utils.data.DataLoader(
		# 					dataset,
		# 					batch_size=256, sampler=val_subsampler)
		model = get_modal(class_names)
		optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
		exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma= gamma)
		criterion = nn.CrossEntropyLoss()
  
		trained_model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc = train(model, criterion, optimizer, exp_lr_scheduler,
																					   train_loader,val_loader,dataset_size,
																					   start_epoch, num_epochs)

		write_results(time_elapsed,training_loss,training_acc,validation_loss,validation_acc,k_folds,num_epochs)
  
		print(F"Completed Fold {fold} of {k_folds}\n")
  
def init_training():
	path = F'~/Results/{dataset}/results_{model_name}.csv'
	open(os.path.expanduser(path),'x')
	path = F'~/Results/{dataset}/time_results_{model_name}.csv'
	open(os.path.expanduser(path),'x')
	path = F'~/Results/{dataset}/hyperparameters_{model_name}.csv'
	open(os.path.expanduser(path),'x')

	#Ensuring the paths for saving modal progress are working

	image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
											data_transforms[x])
					for x in ['train', 'val']}

	class_names = image_datasets['train'].classes
 
 
	
	starting_epoch = 0
	# if load_presaved_model:
	# 	model, optimizer, starting_epoch = load_checkpoint(model, optimizer, os.path.expanduser(model_save_path))
	# else:
	# 	open(os.path.expanduser(model_save_path),'w+')
  
	model = model.to(device)
	write_hyperparameters()
	
	print('Training Started \n\n')
	k_fold(class_names,
			image_datasets,
		 	starting_epoch,max_epochs,k_folds)
 
   
if __name__ == '__main__':
	

	init_training()

	##########################################################---Initializing Model---######################################################################
