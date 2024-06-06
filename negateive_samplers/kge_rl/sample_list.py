#from libc.stdlib cimport rand
from random import randint

#def sample_list(array, int num_samples):
def sample_list(array, num_samples): 
    samples = set()
    arr_len = len(array)
    while True:
        #r = rand() % arr_len
        #r = randint(0,arr_len-1) #ZAFAR IMPORTANT: I might need to hard fix to make sure last element is randomly selected
        
        #ZAFAR: A quick fix of above code can be as follow:
        r = randint(0,arr_len) #ZAFAR IMPORTANT: I might need to hard fix to make sure last element is randomly selected
        if r == arr_len:
            r = r-1 # to make sure index is not out of bound

        samples.add(array[r])
        if len(samples) >= num_samples:
            return list(samples)