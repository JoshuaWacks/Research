from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import csv
import pandas as pd
import seaborn as sns
cudnn.benchmark = True

##########################################################---IMPORTS---############################################################################

chosen_model = 'ResNet18'
training_dataset = 'FairFace_Balanced_Age' 
testing_dataset = 'UTKFace'

training_dataset_path = os.path.join('~/Datasets/FairFace_Balanced_Age',training_dataset)#I made an error when moving files and this dataset is in a folder when the same name
testing_dataset_path = os.path.join('~/Datasets',testing_dataset)

# training_csv_path = '~/Processed_data/balanced_age_FairFace.csv'
training_csv_path = 'Z:/RR/Code/distributions/processed_data/balanced_age_FairFace.csv'
testing_csv_path = '~/Processed_data/processed_UTKFace.csv'

save_path = os.path.join('~/Results,',training_dataset)

model_path = os.path.join(os.path.join('~/Models',training_dataset),chosen_model)

##########################################################---Glabal Variables---######################################################################

def analyse_training_data():
	FairFace_df = pd.read_csv(training_csv_path)
	race_groups = ['Asian','Black','Indian','White']
	race_counts = FairFace_df.race.value_counts().reset_index(name = "count").rename(columns={"index":"race"})
	age_counts = FairFace_df.age.value_counts().reset_index(name = "count").rename(columns={"index":"age"})
	age_counts.sort_values('age',inplace= True)

	gender_counts = FairFace_df.gender.value_counts().reset_index(name = "count").rename(columns={"index":"gender"})

	fig, axs = plt.subplots(1, 3, figsize=(25, 10))
	ax = axs.ravel()

	sns.barplot(x = race_counts.iloc[:, 0], y = race_counts.iloc[:, 1],ax=ax[0])
	ax[0].set_title("Total Counts of Ethnicities in Training Data")


	sns.barplot(x = age_counts.iloc[:, 0], y = age_counts.iloc[:, 1],ax=ax[1])
	ax[1].set_title("Total Counts of Age in Training Data")


	sns.barplot(x = gender_counts.iloc[:, 0], y = gender_counts.iloc[:, 1],ax=ax[2])
	ax[2].set_title("Total Counts of Gender in Training Data")
	plt.savefig(os.path.join(save_path,'Training_Wide_Distribution.png'))
 
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
	plt.savefig(os.path.join(save_path,'Training_Ethnicity_Age_Distribution.png'))

	plt.show()
	
	gender_spread = FairFace_df.groupby('race').gender.value_counts().reset_index(name = 'counts')

	fig, axs = plt.subplots(1, 4, figsize=(25, 10))
	ax = axs.ravel()

	# fig.tight_layout()

	for i,race in enumerate(race_groups):
		t = gender_spread[(gender_spread.race == race)]
		sns.barplot(x = t['gender'], y = t['counts'], ax = ax[i])
		ax[i].set_title(F"{race} Ethnicity")
	plt.savefig(os.path.join(save_path,'Training_Ethnicity_Gender_Distribution.png'))

	plt.show()
		
##########################################################---Analyse Training Data---######################################################################

def analyse_testing_data():
    
	UTKFace_df = pd.read_csv(testing_csv_path)
	UTKFace_df = UTKFace_df[UTKFace_df['new_ethnicity'] != 'latino hispanic']
	UTKFace_df = UTKFace_df[UTKFace_df['new_ethnicity'] != 'middle eastern']

	race_groups = ['asian','black','indian','white']
	race_counts = UTKFace_df.new_ethnicity.value_counts().reset_index(name = "count").rename(columns={"index":"race"})
	age_counts = UTKFace_df.age_group.value_counts().reset_index(name = "count").rename(columns={"index":"age"})
	age_counts.sort_values('age',inplace= True)

	gender_counts = UTKFace_df.gender.value_counts().reset_index(name = "count").rename(columns={"index":"gender"})

	fig, axs = plt.subplots(1, 3, figsize=(25, 10))
	ax = axs.ravel()

	sns.barplot(x = race_counts.iloc[:, 0], y = race_counts.iloc[:, 1],ax=ax[0])
	ax[0].set_title("Total Counts of Ethnicities in Testing Data")


	sns.barplot(x = age_counts.iloc[:, 0], y = age_counts.iloc[:, 1],ax=ax[1])
	ax[1].set_title("Total Counts of Age in Testing Data")


	sns.barplot(x = gender_counts.iloc[:, 0], y = gender_counts.iloc[:, 1],ax=ax[2])
	ax[2].set_title("Total Counts of Gender in Testing Data")
	plt.show()
	plt.savefig(os.path.join(save_path,'Training_Wide_Distribution.png'))
	
 
	age_spread = UTKFace_df.groupby('new_ethnicity').age_group.value_counts().reset_index(name = 'counts')
	age_spread.sort_values('age_group',inplace= True)
	age_spread
	fig, axs = plt.subplots(1, 4, figsize=(25, 10))
	ax = axs.ravel()

	# fig.tight_layout()

	for i,race in enumerate(race_groups):
		t = age_spread[(age_spread.new_ethnicity == race)]
		sns.barplot(x = t['age_group'], y = t['counts'], ax = ax[i])
		ax[i].set_title(F"{race} Ethnicity")

	plt.show()
	plt.savefig(os.path.join(save_path,'Training_Ethnicity_Age_Distribution.png'))
 
	gender_spread = UTKFace_df.groupby('new_ethnicity').gender.value_counts().reset_index(name = 'counts')

	fig, axs = plt.subplots(1, 4, figsize=(25, 10))
	ax = axs.ravel()

	# fig.tight_layout()

	for i,race in enumerate(race_groups):
		t = gender_spread[(gender_spread.new_ethnicity == race)]
		sns.barplot(x = t['gender'], y = t['counts'], ax = ax[i])
		ax[i].set_title(F"{race} Ethnicity")

	plt.show()
	plt.savefig(os.path.join(save_path,'Testing_Ethnicity_Gender_Distribution.png'))

##########################################################---Analyse Testing Data---######################################################################
	
