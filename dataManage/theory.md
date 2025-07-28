# Definition
Linear regression is a linear model, e.g. a model that assumes a linear relationship between the input variables (x) and the single output variable (y). 

More specifically, that output variable (y) can be calculated from a linear combination of the input variables (x).

# Features (variables)

n - number of features
R^(n+1) - vector of n+1 real numbers

# Parameters
Parameters of the hypothesis we want our algorithm to learn in order to be able to do predictions.

# Hypothesis
The equation that gets features and parameters as an input and predicts the value as an output.
For convenience of notation, define X0 = 1.

# Cost Function
Function that shows how accurate the predictions of the hypothesis are with current set of parameters.

xi - input (features) of ith training example
yi - output of ith training example
m - number of training examples

# Gradient Descent
Gradient descent is an iterative optimization algorithm for finding the minimum of a cost function.
To find a local minimum of a function using gradient descent, one takes steps proportional to the negative of the gradient of the function at the current point.

# Feature Scaling
To make linear regression and gradient descent algorithm work correctly we need to make sure that features are on a similar scale.
In order to scale the features we need to do *mean normalization*.







