from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
import time
import os
import copy
import csv

cudnn.benchmark = True
# plt.ion()   # interactive mode


load_presaved_model = False
chosen_model = 'ResNet18'

dataset = 'FairFace_Balanced_Age'

data_dir = os.path.join('~/Datasets/FairFace_Balanced_Age',dataset)

save_path = os.path.join('~/Models',dataset)

save_file = 'model18.pt'
full_path = os.path.join(save_path,save_file)

# Data augmentation and normalization for training
# Just normalization for validation
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
data_transforms = {
	'train': transforms.Compose([ #compose several transforms together
		transforms.RandomResizedCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(mean, std)
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		transforms.Normalize(mean, std) # normalize an image with mean and std
	]),
}

image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
										  data_transforms[x])
				  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
											 shuffle=True, num_workers=4)
			  for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def train_model(model, criterion, optimizer, scheduler,start_epoch, num_epochs=25):
	since = time.time()

	best_model_wts = copy.deepcopy(model.state_dict())
	best_acc = 0.0
 
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
			for inputs, labels in dataloaders[phase]:
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
			else:
				validation_loss.append(epoch_loss)
				validation_acc.append(epoch_acc)

			# deep copy the model
			if phase == 'val' and epoch_acc > best_acc:
				best_acc = epoch_acc
				best_model_wts = copy.deepcopy(model.state_dict())

		print()

	time_elapsed = time.time() - since
	print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
	print(f'Best val Acc: {best_acc:4f}')

	# load best model weights
	model.load_state_dict(best_model_wts)
	return model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc

weights = models.ResNet18_Weights.DEFAULT
model_ft = models.resnet18(weights = weights)
num_ftrs = model_ft.fc.in_features

model_ft.fc = nn.Linear(num_ftrs, len(class_names))
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
starting_epoch = 0

if load_presaved_model :
  model_ft, optimizer, starting_epoch = load_checkpoint(model_ft, optimizer_ft, full_path)


model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=50, gamma=0.1)

num_epochs = 200

trained_model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,starting_epoch,
					   num_epochs=num_epochs)

def write_results_drive(time_elapsed,training_loss,training_acc,validation_loss,validation_acc):
	# file = os.path.join('~/Results/FairFace_Balanced_Age','results.csv')
	path = r'~/Results/FairFace_Balanced_Age/results.csv'
	with open(os.path.expanduser(path),'w') as csvfile:
		fieldnames = ['epoch', 'train_loss','train_acc','val_loss','val_acc']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		for i in range(num_epochs):
			writer.writerow({'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i],'val_loss':validation_loss[i],'val_acc':validation_acc[i]})
   
		writer.writerow({'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i],'val_loss':validation_loss[i],'val_acc':validation_acc[i]})
  
	path = r'~/Results/FairFace_Balanced_Age/time_results.csv'
	with open(os.path.expanduser(path),'w') as csvfile:
		fieldnames = ['model', 'time_taken']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		writer.writerow({'model':chosen_model,'time_taken':time_elapsed})

write_results_drive(time_elapsed,training_loss,training_acc,validation_loss,validation_acc)
