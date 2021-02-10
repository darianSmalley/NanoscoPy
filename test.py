import numpy as np

print('Beginning of the code is here. \n \n')
A = np.array([1,2,3,4,5,6])
B = np.array([1,2,3,4,5,6])
print(A,'\n',B)
keep_elements = (A>=2) & (A<=5)
print(keep_elements)