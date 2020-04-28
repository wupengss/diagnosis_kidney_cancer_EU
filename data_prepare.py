import os
import numpy as np
import pandas as pd
from config_training import config 
import numpy as np
from skimage.measure import label,regionprops
import skimage.morphology as skm
import SimpleITK as sitk
from multiprocessing import Pool
from functools import partial

def lumTrans(img):
    lungwin = np.array([-1200.,600.])
    newimg = (img-lungwin[0])/(lungwin[1]-lungwin[0])
    newimg[newimg<0]=0
    newimg[newimg>1]=1
    newimg = (newimg*255).astype('uint8')
    return newimg

def savenpy_kits19(filelist, kits19_segment, kits19_data,savepath):
    labels_kits = pd.DataFrame({"index":[],"coord":[],"label":[]})
    filelist = list(set(filelist))
    for name in filelist:
        if name == "case00203":
            continue
        else:
            print(name)
            temp = pd.DataFrame({"index":[],"coord":[],"label":[]})

        ## get the file name
            original_file = os.path.join(kits19_data,name+'_imaging.nii')
            segment_file = os.path.join(kits19_segment,name+'_segmentation.nii')

        ## read the segmentation file
            Mask = sitk.ReadImage(segment_file)
            Mask = sitk.GetArrayFromImage(Mask)

        ## remove the kidney annotation from copy
            tumor_region = Mask.copy()
            tumor_region[tumor_region==1]=0

        ## get the region of tumor
            tumor_region = regionprops(tumor_region)

        ## remove the tumor annotation from Mask
            Mask[Mask==2]=0

        ## mophology operation and kideny region extraction
            Mask=skm.binary_closing(Mask)
            Mask=skm.binary_dilation(Mask)
            labels = label(Mask)
            kidney_region = regionprops(labels)

        ## get the index of kidney region 
            Index_kidney = np.argpartition([kidney_region[i].bbox[5]-kidney_region[i].bbox[2] for i in range(len(kidney_region))],-2)[-2:]

        ## extract bounding box
        
            bbox_kidney = [[kidney_region[i].bbox[0], kidney_region[i].bbox[1], kidney_region[i].bbox[2], kidney_region[i].bbox[3],\
            kidney_region[i].bbox[4], kidney_region[i].bbox[5]] for i in Index_kidney]

            bbox_tumor = [[tumor_region[i].bbox[0], tumor_region[i].bbox[1], tumor_region[i].bbox[2], tumor_region[i].bbox[3],\
            tumor_region[i].bbox[4], tumor_region[i].bbox[5]] for i in range(len(tumor_region))]

            bbox_kidney.extend(bbox_tumor)

            temp["index"] = [name for k in range(2+len(tumor_region))]
            temp["coord"] = bbox_kidney
            temp["label"] = ["kidney","kidney"]+["tumor" for i in range(len(tumor_region))]

        ## append the information to dataframe
            labels_kits = pd.concat([labels_kits,temp])

        ## read and save original file
            #origin = sitk.ReadImage(original_file)
            #origin = sitk.GetArrayFromImage(origin)
            #origin_trans = lumTrans(origin)
            #np.save(os.path.join(savepath, name+'_mat.npy'), origin)

    ## save label inforamtion
    labels_kits.to_csv("labels_kits.csv",sep=',',encoding="GB18030",index=False)


def preprocess_kits19():
    kits19_segment = config['kits19_segment']
    savepath = config['preprocess_result_path']
    kits19_data = config['kits19_data']
    kits19_label = config['kits19_label']
    finished_flag = '.flag_preprocesskits19'
    print('starting preprocessing kits19')
    if not os.path.exists(finished_flag):
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        #print('process subset')
        filelist = [f.split('_')[0] for f in os.listdir(kits19_data) if f.endswith('.nii') ]
        if not os.path.exists(savepath):
            os.mkdir(savepath)
        savenpy_kits19(filelist,kits19_segment,kits19_data, savepath)

    print('end preprocessing kits19')
    f= open(finished_flag,"w+")

if __name__=='__main__':
    preprocess_kits19()
