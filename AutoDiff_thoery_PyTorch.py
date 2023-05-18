# resources provided by Jon Krohn, https://github.com/jonkrohn/ML-foundations/blob/master/notebooks/regression-in-pytorch.ipynb

import torch as t
import matplotlib.pyplot as plt

# Goal of Autodiff:
#       Adjust the parameters of the model for the output to be as close as possible to the data.

# Steps:
#       + Forward Propagation: PyTorch builds a computation graph and saves the operations on parameters.

#       + Calculate Loss: It is done to understand the difference between model output and real data. Calculated through Mean Square Error, in this sample script.

#       + Backward Propagation:
#          - PyTorch iterates reversely through the computational graph
#           and calculates the derivative of y with respect to each parameter, i.e Automatic Differentiation.

#          - Calculate gradients of the loss function with respect to the model parameters
#          -  Starts differentiating from the last node(operation) of the graph with respect to the output and works in this way backwards to the first node.
#          - When Loss is calculated in each Epoch (iteration), PyTorch uses these values to calculate derivative of loss
#            with respect to the parameter using the chain rule.

#       + Optimize the parameters: Through Gradient Descent, in this sample script.
#           - Gradients are used to calculate the magnitude of the slope of the ascent or descent of a function.
#           - Therefore Gradient Descent uses the descent (the minimum) of the Loss MSE function.
#           - Gradient Descent is used to lower the Loss score (Cost C in this sample script)
#           - 
#           - Gradient Descent changes values of parameters to bring down the Loss as much as possible in each iteration.
#                This would result in increasing/decreasing a parameter based on the value of the other parameters. Also, if a parameter has a large negative or positive value,
#                   then a corresponding increase or decrease would affect the value of the Loss more than an increase/decrease in other parameters that do not have large values.
#               In this way, the model fits the training data in the best way possible to the model.
#           - Updating Parameters:
#               # θ is the parameter , a is the learning rate(magnitude of change in each parameter optimization iteration),
                # and Loss is the MSE function
#               # θ_i_optimized = θ_i_previous - a*∇Loss(θ_i_previous)


#define linear regression
def linear_regression(m,x,b):
    return m*x+b

# define Mean Square Error
def MSE(y,y_hat):
    # defined as sum of the differences squared of all elements of y hat and y.
    return t.sum((y_hat-y)**2)/len(y)

# set up y data

y = t.tensor([1.86, 1.31, 0.62, 0.33, 0.09, -0.67, -1.23, -1.37])

# set up the b parameter and keep track of how the parameter influences the result of the model
# PyTorch keeps track of all operations performed on b during the forward pass(forward propagation) phase
# Therefore PyTorch calculates derivatives of the loss with respect to b in the backward pass
# Also applying the method will cause the same effect on any other tensor that results from b.
b = t.tensor([0.1]).requires_grad_()

# set up the m parameter

m = t.tensor([0.9]).requires_grad_()

# set up x input for the model(graph)
x_input = t.tensor([i for i in range(8)])

# Forward Propagation
y_hat = linear_regression(m,x_input,b)

# calculate Cost C (Loss)
C = MSE(y,y_hat)

# Backward Propagation
C.backward()

# calculate gradients(derivatives) of m,b
print(m.grad.item())
print(b.grad.item())

# Gradient Descent

# set up Stochastic Gradient Descent (use random subsets of the training data) Optimizer and list parameters b,m
optimizer = t.optim.SGD([m, b], lr=0.01) #lr = learning rate
# SGD is more efficient than GD but may cause noise in parameter optimization
optimizer.step()

# iterative version of the above
epochs = 100 # test 10 iterations

for epoch in range(epochs):

    # reset gradients each iteration as they build up causing incorrect results
    optimizer.zero_grad()

    y_hat = linear_regression(m,x_input,b)
    C = MSE(y,y_hat)
    C.backward()
    optimizer.step() #call autodiff

    print("Epoch = ",epoch," Loss = ", C.item(), " m Gradient = ", m.grad.item(), " b Gradient = ", b.grad.item())

# Since we are only using 8 sample points, we will not get very good approximations as more data points are needed to get desirable approximations.
