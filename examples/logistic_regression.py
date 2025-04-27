#!/usr/bin/python3

# Import Libraries
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Prepare Dataset
# load data
train = pd.read_csv(r"../input/digit_recognizer/train.csv", dtype = np.float32)

# split data into features (pixels) and labels (numbers from 0 to 9)
targets_numpy = train.label.values
features_numpy = train.loc[:,train.columns != "label"].values / 255 # normalization

# train test split. Size of train data is 80% and size of test data is 20%. 
features_train, features_test, targets_train, targets_test = train_test_split(features_numpy,
																			 targets_numpy,
																			 test_size = 0.2,
																			 random_state = 42) 

# create feature and targets tensor for train set. As you remember we need variable to accumulate gradients. Therefore first we create tensor, then we will create variable
featuresTrain = torch.from_numpy(features_train)
targetsTrain = torch.from_numpy(targets_train).type(torch.LongTensor) # data type is long

# create feature and targets tensor for test set.
featuresTest = torch.from_numpy(features_test)
targetsTest = torch.from_numpy(targets_test).type(torch.LongTensor) # data type is long

# batch_size, epoch and iteration
batch_size = 100
n_iters = 10000
num_epochs = n_iters / (len(features_train) / batch_size)
num_epochs = int(num_epochs)

# Pytorch train and test sets
train = torch.utils.data.TensorDataset(featuresTrain, targetsTrain)
test = torch.utils.data.TensorDataset(featuresTest, targetsTest)

# data loader
train_loader = DataLoader(train, batch_size = batch_size, shuffle = False)
test_loader = DataLoader(test, batch_size = batch_size, shuffle = False)

# visualize one of the images in data set
#plt.imshow(features_numpy[10].reshape(28,28))
#plt.axis("off")
#plt.title(str(targets_numpy[10]))
#plt.savefig('graph.png')
#plt.show()


# Create Logistic Regression Model
class LogisticRegressionModel(nn.Module):
	def __init__(self, input_dim, output_dim):
		super(LogisticRegressionModel, self).__init__()
		# Linear part
		self.linear = nn.Linear(input_dim, output_dim)
		# There should be logistic function right?
		# However logistic function in pytorch is in loss function
		# So actually we do not forget to put it, it is only at next parts
	
	def forward(self, x):
		out = self.linear(x)
		return out

# Instantiate Model Class
input_dim = 28*28 # size of image px*px
output_dim = 10  # labels 0,1,2,3,4,5,6,7,8,9

# create logistic regression model
model = LogisticRegressionModel(input_dim, output_dim)

# Cross Entropy Loss  
error = nn.CrossEntropyLoss()

# SGD Optimizer 
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Traning the Model
iteration = 0
loss_list = []
iteration_list = []
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		
		# Define variables
		train = Variable(images.view(-1, 28*28))
		labels = Variable(labels)
		
		# Clear gradients
		optimizer.zero_grad()
		
		# Forward propagation
		outputs = model(train)
		
		# Calculate softmax and cross entropy loss
		loss = error(outputs, labels)
		#print("loss: {}".format(loss))
		
		# Calculate gradients
		loss.backward()
		
		# Update parameters
		optimizer.step()
		
		iteration += 1
		
		# Prediction
		if iteration % 50 == 0:
			# Calculate Accuracy		 
			correct = 0
			total = 0
			# Predict test dataset
			for images, labels in test_loader: 
				test = Variable(images.view(-1, 28*28))
				
				# Forward propagation
				outputs = model(test)
				
				# Get predictions from the maximum value
				predicted = torch.max(outputs.data, 1)[1]
				
				# Total number of labels
				total += len(labels)
				
				# Total correct predictions
				correct += (predicted == labels).sum()
			
			accuracy = 100 * correct / float(total)
			
			# store loss and iteration
			loss_list.append(loss.data)
			iteration_list.append(iteration)
		if iteration % 500 == 0:
			# Print Loss
			print('Iteration: {}  Loss: {}  Accuracy: {}%'.format(iteration, loss.data, accuracy))


# visualization
plt.plot(iteration_list,loss_list)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("Logistic Regression: Loss vs Number of iteration")
plt.show()
