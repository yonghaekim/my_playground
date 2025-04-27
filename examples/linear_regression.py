#!/usr/bin/python3

# Linear Regression with Pytorch

# libraries
import torch	  
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable	 
import torch.nn as nn 
import warnings
warnings.filterwarnings("ignore")

# As a car company we collect this data from previous selling
# lets define car prices
car_prices_array = [3,4,5,6,7,8,9]
car_price_np = np.array(car_prices_array,dtype=np.float32)
car_price_np = car_price_np.reshape(-1,1)
car_price_tensor = Variable(torch.from_numpy(car_price_np))

# lets define number of car sell
number_of_car_sell_array = [ 7.5, 7, 6.5, 6.0, 5.5, 5.0, 4.5]
number_of_car_sell_np = np.array(number_of_car_sell_array,dtype=np.float32)
number_of_car_sell_np = number_of_car_sell_np.reshape(-1,1)
number_of_car_sell_tensor = Variable(torch.from_numpy(number_of_car_sell_np))

# create class
class LinearRegression(nn.Module):
	def __init__(self,input_size,output_size):
		# super function. It inherits from nn.Module and we can access everythink in nn.Module
		super(LinearRegression,self).__init__()
		# Linear function.
		self.linear = nn.Linear(input_dim,output_dim)

	def forward(self,x):
		return self.linear(x)
	
# define model
input_dim = 1
output_dim = 1
model = LinearRegression(input_dim, output_dim) # input and output size are 1

# MSE
mse = nn.MSELoss()

# Optimization (find parameters that minimize error)
learning_rate = 0.02   # how fast we reach best parameters
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)

# train model
loss_list = []
iteration_number = 1001
for iteration in range(iteration_number):
		
	# optimization
	optimizer.zero_grad() 
	
	# Forward to get output
	results = model(car_price_tensor)
	
	# Calculate Loss
	loss = mse(results, number_of_car_sell_tensor)
	
	# backward propagation
	loss.backward()
	
	# Updating parameters
	optimizer.step()
	
	# store loss
	loss_list.append(loss.data)
	
	# print loss
	if(iteration % 50 == 0):
		print('epoch {}, loss {}'.format(iteration, loss.data))

#plt.plot(range(iteration_number),loss_list)
#plt.xlabel("Number of Iterations")
#plt.ylabel("Loss")
#plt.show()

# predict our car price 
predicted = model(car_price_tensor).data.numpy()
plt.scatter(car_prices_array, number_of_car_sell_array, label = "original data", color ="red")
plt.scatter(car_prices_array, predicted, label = "predicted data", color ="blue")

# predict if car price is 10$, what will be the number of car sell
#predicted_10 = model(torch.from_numpy(np.array([10]))).data.numpy()
#plt.scatter(10,predicted_10.data,label = "car price 10$",color ="green")
plt.legend()
plt.xlabel("Car Price $")
plt.ylabel("Number of Car Sell")
plt.title("Original vs Predicted values")
plt.show()
