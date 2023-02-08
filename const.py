import numpy as np
import enum

# Mask used to project the contour of the selected teeth in photos of different views
MASK_UPPER = np.array([ True,  True,  True,  True,  True,  True,  True,   
                  True,  True,  True,  True,  True,  True,  True,  
                  False,  False,  False,  False,  False,  False,  False, 
                  False,  False,  False,  False,  False,  False,  False])

MASK_LOWER = np.array([False,  False,  False,  False,  False,  False,  False,  
                  False,  False,  False,  False,  False,  False,  False, 
                  True,  True,  True,  True,  True,  True,  True,  
                  True,  True,  True,  True,  True,  True,  True])

MASK_LEFT = np.array([ True,  False,  False,  False,  False,  False,  False,   
                  True,  True,  True,  True,  True,  True,  False,  
                  True,  True,  True,  True,  True,  True,  False, 
                  True,  False,  False,  False,  False,  False,  False])

MASK_RIGHT = np.array([ True,  True,  True,  True,  True,  True,  False,   
                  True,  False,  False,  False,  False,  False,  False,  
                  True,  False,  False,  False,  False,  False,  False, 
                  True,  True,  True,  True,  True,  True,  False])

MASK_FRONTAL = np.array([ True,  True,  True,  True,  True,  True,  False,   
                  True,  True,  True,  True,  True,  True,  False,  
                  True,  True,  True,  True,  True,  True,  False, 
                  True,  True,  True,  True,  True,  True,  False])


@enum.unique
class PHOTO(enum.Enum):
    # Enum values must be 0,1,2,3,4 
    UPPER = 0
    LOWER = 1
    LEFT = 2
    RIGHT = 3
    FRONTAL = 4
    
    
NUM_PC = 10 
NUM_POINT = 1500
PG_SHAPE = (NUM_POINT, 3)
UPPER_INDICES = [11,12,13,14,15,16,17,21,22,23,24,25,26,27] # ignore wisdom teeth
LOWER_INDICES = [31,32,33,34,35,36,37,41,42,43,44,45,46,47] # ignore wisdom teeth