import cv2
import numpy as np
import matplotlib.pyplot as plt

######################### input parameters#################
#Define Path of images
img_L = cv2.imread("dataset//umb0.png")  #left
img_R = cv2.imread("dataset//umb1.png")  #right

# calibration mat for cameras left and right
################# enter file name for saving ###################
filename = 'umbrella1.ply'

###### calib bike
'''
calib1 = np.array([[3979.911, 0, 1244.772],
                   [0, 3979.911, 1019.507],
                   [0, 0, 1]])
calib2 = np.array([[3979.911, 0, 1369.115],
                    [0, 3979.911, 1019.507],
                    [0, 0, 1]])

###### calib umbrella
'''
calib1=np.array([[5806.559, 0, 1429.219],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])
calib2=np.array([[5806.559, 0, 1543.51],
                 [0, 5806.559, 993.403],
                 [0, 0, 1]])

######## parameters
win_size = 5
dist1 = np.zeros((1,5)).astype(np.float32)   #distortion coeff
k = calib1
min_disp = 23
max_disp = 245
num_disp = max_disp - min_disp
win_size = 5

### reference link for functional parameters
#https://docs.opencv.org/4.5.1/d2/d85/classcv_1_1StereoSGBM.html

############ tune parameters accordingly
stereo = cv2.StereoSGBM_create(minDisparity = min_disp, numDisparities = num_disp,
                        blockSize = 7,
                        preFilterCap = 1,
                        uniquenessRatio = 2,
                        speckleWindowSize = 50,
                        speckleRange = 2,
                        disp12MaxDiff = 1,
                        P1 = 8*3*win_size**2,
                        P2 = 32*3*win_size**2, mode = 4)

##############################################################################
# tracking features
def track_features(img1,img2):
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    # FINDING INDEX PARAMETERS FOR FLANN OPERATORS
    des1 = np.float32(des1)
    des2 = np.float32(des2)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)
    ########### matching points #########
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    match = flann.knnMatch(des1,des2,k=2)
    matches = flann.knnMatch(des1,des2,k=2)
    pts1 = []
    pts2 = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)
    #print(pts1.shape)
    return pts1,pts2

##### output file format for meshlab
def create_output(vertices, colors, filename):
    colors = colors.reshape(-1,3)
    #colors = np.hstack([colors[:,2],colors[:,1],colors[:,0]]).reshape(-1,3)
    vertices = np.hstack([vertices.reshape(-1,3), colors])
    ply_header = '''ply
        format ascii 1.0
		element vertex %(vert_num)d
		property float x
		property float y
		property float z
		property uchar blue
		property uchar green
		property uchar red
		end_header
    '''
    with open(filename, 'w') as f:
        f.write(ply_header %dict(vert_num = len(vertices)))
        np.savetxt(f, vertices, '%f %f %f %d %d %d')

status = True
if img_L.shape[0] != img_R.shape[0]:
    print("error")
    status = False

while (status):
    ## operation
    # for left and right images
    pts1,pts2 = track_features(img_L,img_R)
    # recover R and T of two images:
    E, mask = cv2.findEssentialMat(pts2,pts1,k,cv2.RANSAC, prob=0.999,threshold = 0.4, mask=None)
    # We select only inlier points
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    #Obtain rotation and translation for the essential matrix
    retval,R,t,mask=cv2.recoverPose(E,pts1,pts2,k)
    size = img_L.shape[0], img_L.shape[1]
    print("recovered pose")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(calib1, dist1, calib2, dist1, size, R, t)
    print("Rectified")
    img_color = cv2.cvtColor(img_L, cv2.COLOR_BGR2RGB)          ### storing color info
    img_L = cv2.cvtColor(img_L, cv2.COLOR_BGR2GRAY )
    img_R = cv2.cvtColor(img_R, cv2.COLOR_BGR2GRAY )
    disparity = stereo.compute(img_L,img_R).astype(np.float32) / num_disp
    ############# filteration block ############
    disparity = cv2.medianBlur(disparity,5)
    print(disparity.shape)
    #print(disparity.shape)
    plt.imshow(disparity,'jet')
    #print("Filtered disparity")
    mask_map = disparity > disparity.min()
    n_cloud = cv2.reprojectImageTo3D(disparity, Q)
    print("nc",n_cloud.shape)
    n_cloud = n_cloud[mask_map]
    out_colors = img_color[mask_map]
    #print("nc",n_cloud.shape)
    create_output(n_cloud, out_colors, filename)
    plt.show()
    print("done")
    status = False
