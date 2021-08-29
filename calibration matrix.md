### Calibration Matrix Bike
calib1 = np.array([[3979.911, 0, 1244.772],
                   [0, 3979.911, 1019.507],
                   [0, 0, 1]])  
                   
calib2 = np.array([[3979.911, 0, 1369.115],
                    [0, 3979.911, 1019.507],
                    [0, 0, 1]])

### Calibration Matrix Umbrella  

calib1 = np.array([[5806.559, 0, 1429.219],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])  
                 
calib2 = np.array([[5806.559, 0, 1543.51],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])

#### Parameters for Bike
win_size = 5  
#distortion coeff  
dist1 = np.zeros((1,5)).astype(np.float32)
k = calib1  
min_disp = 23  
max_disp = 245  
num_disp = max_disp - min_disp  
win_size = 5
