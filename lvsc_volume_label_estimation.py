import pydicom as dicom
#import matplotlib.pylab as plt
#import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import os
#from os.path import isfile, join
#from PIL import Image

import re
import glob
#import nibabel as nib
from imageio import imread#, imsave
from skimage.measure import label

def main():
    # Create dataframe that will be used to store volume estimation labels
    entries = []
    for i in range(40): #max ph allowed is ph39
        temp = [0] * 100
        temp.insert(0, ('ph'+str(i)))
        entries.append(temp)
    # LV end-diastolic volume phase
    temp = [0] * 100
    temp.insert(0, 'LVEDV ph')
    entries.append(temp)
    # LV end-diastolic volume
    temp = [0] * 100
    temp.insert(0, 'LVEDV')
    entries.append(temp)
    # LV end-systolic volume phase
    temp = [0] * 100
    temp.insert(0, 'LVESV ph')
    entries.append(temp)
    # LV end-systolic volume
    temp = [0] * 100
    temp.insert(0, 'LVESV')
    entries.append(temp)

    # Finally create the dataframe with appropriate column names
    columns = []
    for dirlist in os.listdir('CAP_challenge_training_set'):
        if (os.path.isdir('CAP_challenge_training_set/'+dirlist)):
            if (dirlist != '__pycache__'):
                columns.append(dirlist)
    
    columns.insert(0, 'Phase')
    df = pd.DataFrame(entries, columns=columns)

    #howManyOhNo = 0
    c_areas = pd.read_csv("c_areas.csv")

    for directory in os.listdir('CAP_challenge_training_set'):
        print(directory)
        if directory in ['__pycache__','DET0005501','DET0045301','DET0005301','DET0043701']:
            continue
        if (not (os.path.isdir('CAP_challenge_training_set/'+directory))):
            continue

        sax_files = glob.glob(f"CAP_challenge_training_set/{directory}/*SA*.dcm")
        
        sax_slices = set([re.search("SA\d+", _)[0] for _ in sax_files])
        sax_slices = sorted(sax_slices, key=lambda x: int(re.search("\d+", x)[0]))

        sax_dcms = {}
        sax_pngs = {}
        #print(sax_slices)
        for sax_slice in sax_slices:
            sax_slice_files = [_ for _ in sax_files if f"{sax_slice}_" in _]
            sax_dcms[sax_slice] = sorted(sax_slice_files,
            key=lambda x: int(re.search("ph\d+", x)[0].replace("ph","")))
            sax_pngs[sax_slice] = [_.replace(".dcm", ".png") for _ in sax_dcms[sax_slice]]
        

        lvsc_dcm1 = dicom.dcmread(sax_dcms["SA1"][0])

        masks = np.array([[imread(_)[:,:,0] for _ in sax_pngs[_slice]] for _slice in sax_pngs])
        masks[masks > 0] = 1

        maskAreas = np.zeros((masks.shape[0],masks.shape[1]))
        for i in range(masks.shape[0]):
            for j in range(masks.shape[1]):
                img = masks[i,j,:,:]

                imgLbl = label(img, background=1)
                unique, counts = np.unique(imgLbl, return_counts=True)
                if (len(counts) == 3):
                    maskAreas[i,j] = counts[2]
                else:
                    maskAreas[i,j] = 0 # Will attempt to fix/resolve later
                    
        for i in range(maskAreas.shape[0]):
            for j in range(maskAreas.shape[1]):
                # Attempt to estimate the area for slice images that do not have an enclosed LV
                if maskAreas[i,j] == 0:
                    # Check if area was corrected after fixing the "fixable" c-images
                    if (f"{directory}_SA{i+1}_ph{j}.png") in c_areas:
                        if c_areas[f"{directory}_SA{i+1}_ph{j}.png"][0] != 0:
                            maskAreas[i,j] = c_areas[f"{directory}_SA{i+1}_ph{j}.png"][0]
                            continue
                    # If the current area is for either the first or last slice, try to estimate area using
                    # the areas of the same slice from the previous and next time phase.
                    if (i == 0 or i == (maskAreas.shape[0]-1)) and j != 0 and j != (maskAreas.shape[1]-1):
                        if maskAreas[i,j-1] != 0 and maskAreas[i,j+1] != 0:
                            maskAreas[i,j] = (maskAreas[i,j-1] + maskAreas[i,j+1]) / 2
                        # Remain as zero if not possible to find the value. It's not unusual for these slices to be zero anyways
                        continue
                    # Try to estimate current area during the first or last phase by using the slice areas 
                    # on either side of the current slice
                    if (j == 0 or j == (maskAreas.shape[1]-1)) and i != 0 and i != (maskAreas.shape[0]-1):
                        if maskAreas[i-1,j] != 0 and maskAreas[i+1,j] != 0:
                            maskAreas[i,j] = (maskAreas[i-1,j] + maskAreas[i+1,j]) / 2
                            continue
                    # Attempt to estimate the current area using the surrounding slice areas (if they exist)
                    if (j != 0 and j != (maskAreas.shape[1]-1)) and maskAreas[i,j-1] != 0 and maskAreas[i,j+1] != 0: # Try to use images of the same slice at different phases first
                        maskAreas[i,j] = (maskAreas[i,j-1] + maskAreas[i,j+1]) / 2
                        continue
                    elif (i != 0 and i != (maskAreas.shape[0]-1)) and maskAreas[i-1,j] != 0 and maskAreas[i+1,j] != 0: # To to use images on either side of the slice second
                        maskAreas[i,j] = (maskAreas[i-1,j] + maskAreas[i+1,j]) / 2
                        continue

                    # # If maskAreas[i,j] still equals zero, we can leave it as is if it's at either end of the LV
                    # for before in range(i-1,-1,-1):
                    #     if maskAreas[before,j] != 0:
                    #         maskAreas[i,j] = float("inf")
                    #         continue
                    # for after in range(i+1,maskAreas.shape[0]):
                    #     if maskAreas[after,j] != 0:
                    #         maskAreas[i,j] = float("inf")
                    #         continue

                    # If maskAreas[i,j] still equals zero, we can leave it as is if it's at either end of the LV
                    # However, something is likely wrong if the slice is towards the middle of the LV
                    if i > 3 and i < (maskAreas.shape[0]-4):
                        maskAreas[i,j] = float("inf") # A slice this far in shouldn't be zero, "throw" error
                    elif i < 3:
                        if maskAreas[0,j] != 0 or (i == 2 and maskAreas[1,j] != 0) or ((i == 3 or i ==2) and maskAreas[2,j] != 0):
                            maskAreas[i,j] = float("inf") # Not able to estimate this slice towards the front of the LV
                    elif i > (maskAreas.shape[0]-5):
                        if maskAreas[(maskAreas.shape[0]-1),j] != 0 or (i == (maskAreas.shape[0]-3) and maskAreas[(maskAreas.shape[0]-2),j] != 0) or ((i == (maskAreas.shape[0]-4) or i == (maskAreas.shape[0]-3)) and maskAreas[(maskAreas.shape[0]-3),j] != 0):
                            maskAreas[i,j] = float("inf") # Not able to estimate this slice towards the end of the LV
                    # If the current iteration makes it to here, it is not unreasonable that the current slice would have
                    # an area of zero. Hence, maskAreas[i,j] = 0 can remain.

        pixel_spacing_x = lvsc_dcm1.PixelSpacing[0]
        pixel_spacing_y = lvsc_dcm1.PixelSpacing[1]
        pixel_spacing_z = lvsc_dcm1.SliceThickness

        cm_volume_per_pixel = pixel_spacing_x * pixel_spacing_y * pixel_spacing_z * 1e-3

        phaseVolumes = np.sum(maskAreas, axis=0) * cm_volume_per_pixel

        for volIdx in range(phaseVolumes.shape[0]):
            df.loc[volIdx,directory] = phaseVolumes[volIdx]

        # If there are four or more phases where the volume could not be estimated due to poor/missing
        # slice areas, do not set meaningful LVEDV and LVESV values
        unique, counts = np.unique(phaseVolumes, return_counts=True)
        if len(unique) > 1 and np.max(counts) > 3: # Realistically, the volumes of any two phases will not be the same, let alone 4+ phases
            df.loc[40,directory] = float("inf")
            df.loc[41,directory] = float("inf")
            df.loc[42,directory] = float("inf")
            df.loc[43,directory] = float("inf")
        else: # Else set meaningful LVEDV and LVESV values, ignoring any "inf" so long as there are only a couple
            phaseVolumes[phaseVolumes == float("inf")] = np.mean(phaseVolumes[phaseVolumes != float("inf")]) # Will ensure these phases are not selected
            df.loc[40,directory] = np.argmax(phaseVolumes)
            df.loc[41,directory] = phaseVolumes[np.argmax(phaseVolumes)]
            df.loc[42,directory] = np.argmin(phaseVolumes)
            df.loc[43,directory] = phaseVolumes[np.argmin(phaseVolumes)]

    print(df)            
    df.to_csv('CAP_challenge_training_set/volume_labels_resolved_c_shapes.csv', index = False, header=True)
    #print(f"There were a total of {howManyOhNo} 'Oh no!'s")

if __name__ == "__main__":
    main()
