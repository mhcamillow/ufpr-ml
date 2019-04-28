# import numpy as np
# import math

# a = np.array([i for i in range(200)])
# print(a)

# threads = 11
# thread_size = math.ceil(len(a) / (threads))

# print("Thread size: " + str(thread_size))

# for x in range(threads):
#     first = x * thread_size
#     last = (x + 1) * thread_size - 1
#     last = len(a) - 1 if (x == threads - 1) else last
#     print('First: ' + str(a[first]))
#     print('Last: ' + str(a[last]))


from multiprocessing import Process, Array

predicted = Array('i', range(100))

print(predicted[0:101])