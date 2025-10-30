import numpy as np
import pandas as pd

def gradient_descent(x,y):
    m_curr = b_curr = 0
    iterations = 100000
    n = len(x)
    learning_rate = 0.08
    for i in range(iterations):
        y_predicted = (m_curr*x) + b_curr # Predicted y values
        cost = (1/n)*sum((y - y_predicted)**2) # Cost function
        md = (-2/n)*sum(x*(y-y_predicted)) # Derivative of m
        bd = (-2/n)*sum(y-y_predicted) # Derivative of b
        m_curr = m_curr - learning_rate*md # Update m
        b_curr = b_curr - learning_rate*bd # Update b
        print(f"m: {m_curr}, b: {b_curr}, iteration: {i}, cost: {cost}")


x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])
#gradient_descent(x,y)

#Exercise is to come up with a linear function for given test results using gradient descent

df = pd.read_csv("Day3_Gradient_Descent/test_scores.csv")
math_grade = np.array(df.math)
cs_grade = np.array(df.cs)
gradient_descent(math_grade, cs_grade)