from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
import matplotlib.pyplot as plt
import cv2

origin_file = r'case00004_imaging.nii'
seg_file = r'case00004_segmentation.nii'
dcm_tumor = r'G:\data\3Dircadb\3Dircadb1.3\MASKS_DICOM\livertumor\image_137'


def drawContour():

    image = sitk.ReadImage(origin_file)
    image_array = sitk.GetArrayFromImage(image)
    #image_array = np.squeeze(image_array)
    image_array = image_array.astype(np.float32)
	# windowing 操作
    # min:-200, max:200
    # img = (img-min)/(max - min)
    image_array = (image_array - (-200)) / 400.0
    image_array[image_array > 1] = 1.0
    image_array[image_array < 0] = 0.0

    image_array = [cv2.cvtColor(image_array[:,:,x], cv2.COLOR_GRAY2BGR) for x in range(64)]
    image_array = np.array(image_array)
    kidney = sitk.ReadImage(seg_file)
    kidney_array = sitk.GetArrayFromImage(kidney)

    #xx,yy= np.where(kidney_array[:,:,25]==1)
    #box = np.array([[np.min(xx),np.min(yy)],[np.max(xx),np.max(yy)]])

    contours, hierarchy = cv2.findContours(kidney_array[:,:,25], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_array[25,:,:],contours,-1,(255, 0, 0))
    for contour in contours:
        x,y,w,h = cv2.boundingRect(contour)
        cv2.rectangle(image_array[25,:,:], (x,y), (x+w,y+h), (0,255,0), 1)
    #contours, hierarchy = cv2.findContours(kidney_array[:,:,25], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #box_coord = cv2.minAreaRect(kidney_array[:,:,25])
    
    #cv2.rectangle(image_array[25,:,:], (box[0][1],box[0][0]), (box[1][1],box[1][0]), (0,255,0))
    cv2.imshow("kidney_contour",image_array[25,:,:])
    cv2.waitKey()
    
    plt.imshow(kidney_array[:,:,25],cmap='gray')
    plt.show()

#def extract_rect_coord(img):


if __name__=='__main__':
    drawContour()




