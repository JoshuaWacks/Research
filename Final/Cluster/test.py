import csv
import os


path = r'~/Results/FairFace_Balanced_Age/test.csv'
file = os.path.expanduser(path)
print(file)
with open(file,'w') as csvfile:
	print("OPEN")
	fieldnames = ['H1']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
 
	writer.writerow({'H1':'Test'})
	