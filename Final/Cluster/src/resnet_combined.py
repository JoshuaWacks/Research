from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import time
import os
import copy
import csv

cudnn.benchmark = True
##########################################################---IMPORTS---############################################################################
chosen_model = 'ResNet18'

load_presaved_model = True

age_training_dataset = 'FairFace_Balanced_Age' 
training_dataset_path = os.path.join('~/Datasets/FairFace_Balanced_Age',age_training_dataset)#I made an error when moving files and this dataset is in a folder when the same name
age_weights_file = 'model18_A_13_SS_100.pt'
model_age_path = os.path.join(os.path.join('~/Models',age_training_dataset),age_weights_file)

ethnicity_training_dataset = 'FairFace_ethnicity_training' 
ethnicity_weights_file = 'model18.pt'
model_ethnicity_path = os.path.join(os.path.join('~/Models',ethnicity_training_dataset),ethnicity_weights_file)

model_name = 'model18_Combined_A_13_SS_100'#Alpha: 1e_3, Step_size: 100, before we decrease alpha
model_save_path = os.path.join(os.path.join('~/Models',age_training_dataset),model_name+'.pt')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
##########################################################---Glabal Variables---######################################################################
def load_checkpoint(model, optimizer, load_path):
	if torch.cuda.is_available():
		
		checkpoint = torch.load(load_path)
	else:
		checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
		
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']

	return model, optimizer, epoch

def save_checkpoint(model, optimizer, save_path, epoch):
	torch.save({
		'model_state_dict': model.state_dict(),
		'optimizer_state_dict': optimizer.state_dict(),
		'epoch': epoch
	}, save_path)

def get_model_data():
	image_datasets = {x: datasets.ImageFolder(os.path.join(training_dataset_path, x),
										  data_transforms[x])
				  for x in ['train', 'val']}
	dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50,
												shuffle=True, num_workers=4)
				for x in ['train', 'val']}
	dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
	class_names = image_datasets['train'].classes
 
	weights = models.ResNet18_Weights.DEFAULT
	model_ft = models.resnet18(weights = weights)
	num_ftrs = model_ft.fc.in_features

	model_ft.fc = nn.Linear(num_ftrs,len(class_names))
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
 
	model_age, optimizer, starting_epoch = load_checkpoint(model_ft, optimizer_ft, os.path.expanduser(model_age_path))
	model_age = model_age.to(device)
	
	model_ft.fc = nn.Linear(num_ftrs,4)
	model_ethnicity, optimizer, starting_epoch = load_checkpoint(model_ft, optimizer_ft, os.path.expanduser(model_ethnicity_path))
	model_ethnicity = model_ethnicity.to(device)
 
	return model_age,model_ethnicity,dataloaders,class_names,dataset_sizes

##########################################################---Loading In File Data and Model Data---######################################################################

class Identity(nn.Module):
	def __init__(self):
		super(Identity, self).__init__()
		
	def forward(self, x):
		return x
	
class MyEnsemble(nn.Module):
	def __init__(self, modelA, modelB,num_classes):
		super(MyEnsemble, self).__init__()
		# self.modelA =  torch.nn.Sequential(*(list(modelA.children())[:-1]))
		# self.modelB = torch.nn.Sequential(*(list(modelB.children())[:-1]))
		self.modelA = modelA
		self.modelB = modelB

		self.modelA.fc = Identity()
		self.modelB.fc = Identity()
	
		# for param in self.modelA.parameters():
		# 	param.requires_grad = False
	
		# for param in self.modelB.parameters():
		# 	param.requires_grad = False
   
		self.classifier = nn.Linear((512+512), num_classes)
		
	def forward(self, x1):
		x1_out = self.modelA(x1)
		x2_out = self.modelB(x1)
  
		x = torch.cat((x1_out, x2_out), dim=1)
		x = self.classifier(nn.functional.softmax(x,dim = 0))
		return x
	
def build_combined_model(model_age,model_ethnicity,class_names):
	combined_model = MyEnsemble(model_age,model_ethnicity,len(class_names))
 
	return combined_model.to(device)

##########################################################---Building Combined Model---######################################################################

def train_model(model, criterion, optimizer, scheduler,start_epoch, dataset_sizes,num_epochs=25):
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
    
			save_checkpoint(model, optimizer, os.path.expanduser(model_save_path),epoch)

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

##########################################################---Function To Train Model---######################################################################

def write_results_drive(time_elapsed,training_loss,training_acc,validation_loss,validation_acc):
	# file = os.path.join('~/Results/FairFace_Balanced_Age','results.csv')
	file_name = F'time_results+{model_name}.csv'
	path = F'~/Results/FairFace_Balanced_Age/{file_name}'
	with open(os.path.expanduser(path),'w+') as csvfile:
		fieldnames = ['epoch', 'train_loss','train_acc','val_loss','val_acc']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		for i in range(num_epochs):
			writer.writerow({'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i],'val_loss':validation_loss[i],'val_acc':validation_acc[i]})
   
		writer.writerow({'epoch':i, 'train_loss':training_loss[i],'train_acc':training_acc[i],'val_loss':validation_loss[i],'val_acc':validation_acc[i]})
  
	file_name = F'time_results+{model_name}.csv'
	path = F'~/Results/FairFace_Balanced_Age/{file_name}'
	with open(os.path.expanduser(path),'w+') as csvfile:
		fieldnames = ['model', 'time_taken']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
  
		writer.writerow({'model':chosen_model,'time_taken':time_elapsed})
  
  
model_age,model_ethnicity,dataloaders,class_names,dataset_sizes = get_model_data()

combined_model = build_combined_model(model_age,model_ethnicity,class_names)

optimizer_ft = optim.Adam(combined_model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=100, gamma=0.05)
starting_epoch = 0

if os.path.exists(os.path.expanduser(model_save_path)):
	if load_presaved_model:
		model_ft, optimizer, starting_epoch = load_checkpoint(combined_model, optimizer_ft, os.path.expanduser(model_save_path))
else:
    open(os.path.expanduser(model_save_path),'w+')

num_epochs = 250

model,time_elapsed,training_loss,training_acc,validation_loss,validation_acc = train_model(combined_model,criterion,optimizer_ft,exp_lr_scheduler,starting_epoch,dataset_sizes, num_epochs=num_epochs)

write_results_drive(time_elapsed,training_loss,training_acc,validation_loss,validation_acc)


##########################################################---Initializing Model---######################################################################

