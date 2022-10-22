from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import os
import csv
import pandas as pd
import seaborn as sns
cudnn.benchmark = True
import sklearn.metrics as metrics
from PIL import Image
from prettytable import PrettyTable

##########################################################---IMPORTS---############################################################################

training_dataset = 'FairFace_Balanced_Age' 
testing_dataset = 'FairFace_Balanced_Age'

training_dataset_path = os.path.join('~/Datasets/FairFace_Balanced_Age',training_dataset)#I made an error when moving files and this dataset is in a folder when the same name
testing_dataset_path = os.path.join('~/Datasets/FairFace_Balanced_Age',testing_dataset)

training_csv_path = '~/processed_data/FairFace_train_val.csv'
testing_csv_path = '~/processed_data/FairFace_test.csv'

model_name= 'ResNet50_No_Normalizing_A_13_SS_25_batch_32_epochs_100'
weights_file = F'{model_name}.pt'
model_path = os.path.join(os.path.join('~/Models',training_dataset),weights_file)

results_file = F'results_{model_name}.csv'

output_folder = F'output/FairFace_Balanced_Age_Results/{model_name}'
csv_results_output_path = os.path.join(os.path.join('~/src',output_folder),F'{model_name}.csv')

if not os.path.exists(output_folder): os.makedirs(output_folder)
if not os.path.exists(os.path.expanduser(csv_results_output_path)): open(os.path.expanduser(csv_results_output_path),'x')

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
		# transforms.Normalize(mean, std)
	]),
	'val': transforms.Compose([
		transforms.Resize(256),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std) # normalize an image with mean and std
	]),
 	'test': transforms.Compose([
		transforms.Resize(256),
		# transforms.CenterCrop(224),
		transforms.ToTensor(),
		# transforms.Normalize(mean, std) # normalize an image with mean and std
	])
}

##########################################################---Glabal Variables---######################################################################

def analyse_data(dataset,split):
	FairFace_df = dataset
 
	race_groups = ['Asian','Black','Indian','White']
	race_counts = FairFace_df.race.value_counts().reset_index(name = "count").rename(columns={"index":"race"})
	age_counts = FairFace_df.age.value_counts().reset_index(name = "count").rename(columns={"index":"age"})
	age_counts.sort_values('age',inplace= True)

	gender_counts = FairFace_df.gender.value_counts().reset_index(name = "count").rename(columns={"index":"gender"})

	fig, axs = plt.subplots(1, 3, figsize=(25, 10))
	ax = axs.ravel()

	sns.barplot(x = race_counts.iloc[:, 0], y = race_counts.iloc[:, 1],ax=ax[0])
	ax[0].set_title(F"Total Counts of Ethnicities in {split} Data")


	sns.barplot(x = age_counts.iloc[:, 0], y = age_counts.iloc[:, 1],ax=ax[1])
	ax[1].set_title(F"Total Counts of Age in {split} Data")


	sns.barplot(x = gender_counts.iloc[:, 0], y = gender_counts.iloc[:, 1],ax=ax[2])
	ax[2].set_title(F"Total Counts of Gender in {split} Data")
 
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, F'{split}_Wide_Distribution.png')
	plt.savefig(file_path)

	plt.show()
#Done training wide data

	age_spread = FairFace_df.groupby('race').age.value_counts().reset_index(name = 'counts')
	age_spread.sort_values('age',inplace= True)

	fig, axs = plt.subplots(1, 4, figsize=(25, 10))
	ax = axs.ravel()

	# fig.tight_layout()

	for i,race in enumerate(race_groups):
		t = age_spread[(age_spread.race == race)]
		sns.barplot(x = t['age'], y = t['counts'], ax = ax[i])
		ax[i].set_title(F"{race} Ethnicity")
  
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, F'{split}_Ethnicity_Age_Distribution.png')
	plt.savefig(file_path)
	
	gender_spread = FairFace_df.groupby('race').gender.value_counts().reset_index(name = 'counts')

	fig, axs = plt.subplots(1, 4, figsize=(25, 10))
	ax = axs.ravel()

	# fig.tight_layout()

	for i,race in enumerate(race_groups):
		t = gender_spread[(gender_spread.race == race)]
		sns.barplot(x = t['gender'], y = t['counts'], ax = ax[i])
		ax[i].set_title(F"{race} Ethnicity")
	  
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, F'{split}_Ethnicity_Gender_Distribution.png')
	plt.savefig(file_path)
	

	plt.show()
		
##########################################################---Analyse Data---######################################################################
	
def save_training_graphs():
	results_path = os.path.join(os.path.join('~/Results',training_dataset),results_file)
	results_df = pd.read_csv(results_path)
	results_df['train_acc'] = results_df['train_acc'].astype(str).str[7:13]
	results_df['val_acc'] = results_df['val_acc'].astype(str).str[7:13]
 
	plt.figure(figsize=(15,10))
	_ = sns.lineplot(data = results_df, x = 'epoch', y = 'train_loss')
	plt.title(F'Train loss over the {training_dataset} dataset vs epoch')
	plt.show()

	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Training_Loss.png')
	plt.savefig(file_path)
	plt.clf()
 
	_ = sns.lineplot(data = results_df, x = 'epoch', y = 'val_loss')
	plt.title(F'Val loss over the {training_dataset} dataset vs epoch')
	plt.show()
 
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Validation_Loss.png')
	plt.savefig(file_path)
	plt.clf()
 

	plot = sns.lineplot(data = results_df, x = 'epoch', y = 'train_acc')
	plot.invert_yaxis()
	plot.set(yticklabels=[])
	plt.title(F'Training Accuracy over the {training_dataset} dataset vs epoch')
	plt.show()

	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Training_Accuracy.png')
	plt.savefig(file_path)
	plt.clf()

	plot = sns.lineplot(data = results_df, x = 'epoch', y = 'val_acc')
	plot.invert_yaxis()
	plot.set(yticklabels=[])
	plt.title(F'Validation Accuracy over the {training_dataset} dataset vs epoch')
	plt.show()
	
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Validation_Accuracy.png')
	plt.savefig(file_path)
	plt.clf()
 
	_ = sns.lineplot(data = results_df, x = 'epoch', y = 'train_loss')
	_ = sns.lineplot(data = results_df, x = 'epoch', y = 'val_loss')
	plt.title(F'Loss vs epoch')
	plt.legend(title= 'Split',labels = ['Training','Validation'])
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Overall_Loss.png')
	plt.savefig(file_path)
	plt.clf()
 
	plot = sns.lineplot(data = results_df, x = 'epoch', y = 'train_acc')
	plot.invert_yaxis()
	plot.set(yticklabels=[])
	plot = sns.lineplot(data = results_df, x = 'epoch', y = 'val_acc')
	plt.title(F'Accuracy vs epoch')
	plt.legend(title= 'Split',labels = ['Training','Validation'])
	plot.invert_yaxis()
	plot.set(yticklabels=[])
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, 'Overall_Accuracy.png')
	plt.savefig(file_path)
	plt.clf()
 
 
##########################################################---Training Graphs---######################################################################

def load_checkpoint(model, optimizer, load_path):
	if torch.cuda.is_available():
		
		checkpoint = torch.load(load_path)
	else:
		checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
		
	model.load_state_dict(checkpoint['model_state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	epoch = checkpoint['epoch']

	return model, optimizer, epoch

def get_model_data():
	image_datasets_validation = {x: datasets.ImageFolder(os.path.join(training_dataset_path, x),
											data_transforms[x])
					for x in ['val']}
	dataloaders_validation = {x: torch.utils.data.DataLoader(image_datasets_validation[x], batch_size=4,
												shuffle=True, num_workers=4)
				for x in ['val']}	
 
	image_datasets_testing = {x: datasets.ImageFolder(os.path.join(testing_dataset_path, x),
											data_transforms[x])
					for x in ['test']}
	dataloaders_testing = {x: torch.utils.data.DataLoader(image_datasets_testing[x], batch_size=4,
												shuffle=True, num_workers=4)
				for x in ['test']}

 
	class_names = image_datasets_validation['val'].classes
 
	weights = models.ResNet34_Weights.DEFAULT
	model_ft = models.resnet34(weights = weights)
	num_ftrs = model_ft.fc.in_features

	model_ft.fc = nn.Linear(num_ftrs,len(class_names))
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
 
	model_ft, optimizer, starting_epoch = load_checkpoint(model_ft, optimizer_ft, os.path.expanduser(model_path))

	model_ft = model_ft.to(device)
	model_ft.eval()
	
	return model_ft,dataloaders_validation,dataloaders_testing,class_names
##########################################################---Loading In File Data and Model Data---######################################################################

def get_predictions(model,data,data_name,one_off = False):
 
	predictions = []
	true_labels = []
	
	for inputs, labels in data[data_name]:
		inputs = inputs.to(device)
		labels = labels.to(device)

		with torch.set_grad_enabled(False):
			outputs = model(inputs)
			_, preds = torch.max(outputs, 1)
   
		for true,pred in zip(labels.cpu().numpy(),preds.cpu().numpy()):
	  
			true_labels.append(true)
   
			if one_off:
				if true == (pred-1):
					predictions.append((pred-1))
				elif true == (pred+1):
					predictions.append((pred+1))
				else:
					predictions.append(pred)
					
			else:
				predictions.append(pred)
			
	return true_labels,predictions
	
	 
def write_results(true,predicted,field):
	
	accuracy = metrics.accuracy_score(true,predicted)
	balanced_accuracy = metrics.balanced_accuracy_score(true,predicted)
 
##########################################################---Accuracy---######################################################################
	
	precision_macro = metrics.precision_score(true,predicted,average='macro')
	precision_micro = metrics.precision_score(true,predicted,average='micro')
	precision_weighted = metrics.precision_score(true,predicted,average='weighted')
 
##########################################################---Precision--######################################################################
	
	recall_macro = metrics.recall_score(true,predicted,average='macro')
	recall_micro = metrics.recall_score(true,predicted,average='micro')
	recall_weighted = metrics.recall_score(true,predicted,average='weighted')
 
##########################################################---Recall--######################################################################
 
	f1_macro = metrics.f1_score(true,predicted,average='macro')
	f1_micro = metrics.f1_score(true,predicted,average='micro')
	f1_weighted = metrics.f1_score(true,predicted,average='weighted')
 
##########################################################---F1 Score---######################################################################

	cohen_kappa = metrics.cohen_kappa_score(true,predicted)

	field_names = ['Metric', 'Result']
	
	with open(os.path.expanduser(csv_results_output_path),'a') as csvfile:
		writer = csv.DictWriter(csvfile, fieldnames=field_names)
	
		writer.writerow({'Metric':F"accuracy {field}",'Result':accuracy})
		writer.writerow({'Metric':F"balanced_accuracy {field}",'Result':balanced_accuracy})
  
		writer.writerow({'Metric':F"precision_macro {field}",'Result':precision_macro})
		writer.writerow({'Metric':F"precision_micro {field}",'Result':precision_micro})
		writer.writerow({'Metric':F"precision_weighted {field}",'Result':precision_weighted})
  
		writer.writerow({'Metric':F"recall_macro {field}",'Result':recall_macro})
		writer.writerow({'Metric':F"recall_micro {field}",'Result':recall_micro})
		writer.writerow({'Metric':F"recall_weighted {field}",'Result':recall_weighted})
  
		writer.writerow({'Metric':F"f1_macro {field}",'Result':f1_macro})
		writer.writerow({'Metric':F"f1_micro {field}",'Result':f1_micro})
		writer.writerow({'Metric':F"f1_weighted {field}",'Result':f1_weighted})
  
		writer.writerow({'Metric':F"cohen_kappa {field}",'Result':cohen_kappa})

  
##########################################################---Writing to CSV File--######################################################################
  
  
	table = PrettyTable()
	table.field_names = field_names
 
	table.add_row([F"accuracy {field}",accuracy])
	table.add_row([F"balanced_accuracy {field}",balanced_accuracy])
 
	table.add_row([F"precision_macro {field}",precision_macro])
	table.add_row([F"precision_micro {field}",precision_micro])
	table.add_row([F"precision_weighted {field}",precision_weighted])
 
	table.add_row([F"recall_macro {field}",recall_macro])
	table.add_row([F"recall_micro {field}",recall_micro])
	table.add_row([F"recall_weighted {field}",recall_weighted])
 
	table.add_row([F"f1_macro {field}",f1_macro])
	table.add_row([F"f1_micro {field}",f1_micro])
	table.add_row([F"f1_weighted {field}",f1_weighted])
 
	table.add_row([F"cohen_kappa {field}",cohen_kappa])
 
 
	print(table)
	string_table = table.get_string();
 
	txt_file = open(F'{output_folder}/{field}.txt','a')
	txt_file.write(string_table)
	txt_file.close()
 
##########################################################---Displaying table of data--######################################################################
 

def get_confusion(true,predicted,field):
	cf_matrix = metrics.confusion_matrix(true,predicted)
	sns.heatmap(cf_matrix,annot=True)
 
	current_directory_path = os.getcwd()
	subfolder_path = os.path.join(current_directory_path, output_folder)
	file_path = os.path.join(subfolder_path, F'{field}_Confusion_Matrix.png')
	plt.savefig(file_path)
	plt.clf()
 
	sns.heatmap(cf_matrix/np.sum(cf_matrix),annot=True,fmt='.2%', cmap='Blues')
	file_path = os.path.join(subfolder_path, F'{field}_Percentage_Confusion_Matrix.png')
	plt.savefig(file_path)
	plt.clf()
	

def obtain_overall_results(model,dataloaders_validation,dataloaders_testing):
	

	true_labels_validation,predictions_validation = get_predictions(model,dataloaders_validation,'val',one_off=False)
	true_labels_testing,predictions_testing = get_predictions(model,dataloaders_testing,'test',one_off=False)
 
	true_labels_validation_one_off,predictions_validation_one_off = get_predictions(model,dataloaders_validation,'val',one_off=True)
	true_labels_testing_one_off,predictions_testing_one_off = get_predictions(model,dataloaders_testing,'test',one_off=True)
	
 
	with open(os.path.expanduser(csv_results_output_path),'w') as csvfile:
		fieldnames = ['Metric', 'Result']
		writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
		writer.writeheader()
 
	write_results(true_labels_validation,predictions_validation,"Validation")
	get_confusion(true_labels_validation,predictions_validation,"Validation")
 
	write_results(true_labels_testing,predictions_testing,"Testing")
	get_confusion(true_labels_testing,predictions_testing,"Testing")
 
	write_results(true_labels_validation_one_off,predictions_validation_one_off,"Validation_One_Off")
	get_confusion(true_labels_validation_one_off,predictions_validation_one_off,"Validation_One_Off")
 
	write_results(true_labels_testing_one_off,predictions_testing_one_off,"Testing_One_Off")
	get_confusion(true_labels_testing_one_off,predictions_testing_one_off,"Testing_One_Off")
 
	
def get_pred(model,file):
	img = Image.open(file)
	input = data_transforms['val'](img)
	input = input.unsqueeze(0)
	input = input.to(device)
	with torch.set_grad_enabled(False):
		output = model(input)
		_, preds = torch.max(output, 1)
  
	return preds[0].item() 

def get_ethnicity_predictions(model,class_names,one_off = False):
	
	predictions = {'white': [],'black': [],'indian': [],'asian': []}
	true_labels = {'white': [],'black': [],'indian': [],'asian': []}
 
	FairFace_testing_df = pd.read_csv(testing_csv_path)
	for i in range(len(FairFace_testing_df)):
		eth = FairFace_testing_df.iloc[i]['race']
		if eth == 'Others':
			continue
		age = FairFace_testing_df.iloc[i]['age']
		if  age == 'more_than_70':
			true = 8

		elif age == '0-2':
			true = 0	
		elif age == '3-9':
			true = 1
		else:
			true = class_names.index(age)
	
		name = FairFace_testing_df.iloc[i]['file']
		file_name = name.split('/')[-1]

		pre_path = os.path.join(os.path.expanduser('~/Datasets/FairFace_Balanced_Age/FairFace_Balanced_Age/test'),age)
		image_path = os.path.join(pre_path,file_name)
		pred = get_pred(model,image_path)

		eth = eth.lower()
		true_labels[eth].append(true)
  
		if one_off:
			if true == (pred-1):
				predictions[eth].append((pred-1))
			elif true == (pred+1):
				predictions[eth].append((pred+1))
			else:
				predictions[eth].append(pred)
				
		else:
			predictions[eth].append(pred)
   
	return true_labels,predictions

def obtain_ethnicity_results(model,class_names):
	true_labels,predictions = get_ethnicity_predictions(model,class_names)
	true_labels,predictions_one_off = get_ethnicity_predictions(model,class_names,one_off=True)
 
 
	for eth in ['white','black','indian','asian']:
		write_results(true_labels[eth],predictions[eth],F"Testing_{eth}")
		get_confusion(true_labels[eth],predictions[eth],F"Testing_{eth}")
  
		write_results(true_labels[eth],predictions_one_off[eth],F"Testing_{eth}_One_Off")
		get_confusion(true_labels[eth],predictions_one_off[eth],F"Testing_{eth}_One_Off")
  
# FairFace_df = pd.read_csv(training_csv_path)
# FairFace_df = FairFace_df[FairFace_df['split'] == 'train']
# analyse_data(FairFace_df,'Training')

# FairFace_testing_df = pd.read_csv(testing_csv_path)
# analyse_data(FairFace_testing_df,'Testing')

save_training_graphs()

# model,dataloaders_validation,dataloaders_testing,class_names = get_model_data()
# print('Running Analysis')
# obtain_overall_results(model,dataloaders_validation,dataloaders_testing)
# obtain_ethnicity_results(model,class_names)