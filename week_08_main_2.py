# Import your packages, modules, global constants first
from self_py_fun.week_08_example_fun import * #call all the functions from self written --> uploads automatically to main funciton in other window, should be able to use functions without having to re-import??
import numpy as np #ones we already have downloaded 

# Call your variable
print(alpha)

# Call your functions
# Since you import everything, we do not have to write ef.xxx
print(message_hello('Tianwen'))
print(fn_cubic(3))
print(np.array([1.0]))

random()
np.random