import torch
import numpy as np
from itertools import islice 
  
  
# Slicing the range function 
for i in islice(range(20), 5):  
    print(i) 
      
      
li = [2, 4, 5, 7, 8, 10, 20]  
  
# Slicing the list 
print(list(islice(li,3)))   

a = torch.rand(3,5)
print(a)
print(a[a>0.3])
print(torch.pow(a[a>0.3], 0.1))