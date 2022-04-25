import os

import cv2
import numpy as np
import torch
import torch.utils.data

#__all__ = ['SegmentationDataset', 'VolumeDataset', 'GetMasksDataset']
__all__ = ['SegmentationDataset', 'VolumeDataset']


class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
        """
        Args:
            img_ids (list): Image ids.
            img_dir: Image file directory.
            mask_dir: Mask file directory.
            img_ext (str): Image file extension.
            mask_ext (str): Mask file extension.
            num_classes (int): Number of classes.
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.png
            │   ├── 0aab0a.png
            │   ├── 0b1761.png
            │   ├── ...
            |
            └── masks
                ├── 0
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                |
                ├── 1
                |   ├── 0a7e06.png
                |   ├── 0aab0a.png
                |   ├── 0b1761.png
                |   ├── ...
                ...
        """
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_ext = img_ext
        self.mask_ext = mask_ext
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        
        img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

        mask = []
        for i in range(self.num_classes):
            mask.append(cv2.imread(os.path.join(self.mask_dir, str(i),
                        img_id + self.mask_ext), cv2.IMREAD_GRAYSCALE)[..., None])
        mask = np.dstack(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)
        
        return img, mask, {'img_id': img_id}


class VolumeDataset(torch.utils.data.Dataset):
    def __init__(self, patient_data, patient_dir, img_ext, numShortSlices, yesLongAxis, transform=None):
        """
        Args:
            short_img_ids (list): Short axis image ids.
            long_img_ids (list): Long axis image ids.
            img_dir: Image file directory.
            img_ext (str): Image file extension.
            num_classes (int): Number of classes.
            volume (dec): Current patient's LV volume
            transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
        Note:
            Make sure to put the files as the following structure:
            <dataset name>
            ├── images
            |   ├── 0a7e06.png
            │   ├── 0aab0a.png
            │   ├── 0b1761.png
            │   ├── ...
            ...
        """
        self.patient_data = patient_data
        self.patient_dir = patient_dir
        self.img_ext = img_ext
        self.numShortSlices = numShortSlices
        self.yesLongAxis = yesLongAxis
        self.transform = transform

    def __len__(self):
        return len(self.patient_data)

    def __getitem__(self, idx):
        patients_info = self.patient_data[idx]
        patient = "pat_" + str(int(patients_info[0]))
        volumes = [patients_info[2], patients_info[1]]

        all_imgs = os.listdir(os.path.join(self.patient_dir, patient))

        long_imgs = []
        short_imgs = []

        if self.yesLongAxis:
            long_imgs_path = list(filter(lambda x: 'lax' in x, all_imgs))
            for long_img in long_imgs_path:
                img = cv2.imread(os.path.join(self.patient_dir, patient, long_img))

                if self.transform is not None:
                    augmented = self.transform(image=img)
                    img = augmented['image']
                
                img = img.astype('float32') / 255
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)

                long_imgs.append(img)

        short_imgs_paths = list(filter(lambda x: 'sax' in x, all_imgs))
        num_phases = len(all_imgs) - len(short_imgs_paths)
        num_slices = len(short_imgs_paths) / num_phases
        
        short_slices_to_use = []
        if self.numShortSlices == 1:
            short_slices_to_use = [int(1 + num_slices/2)]
        elif self.numShortSlices == 3:
            short_slices_to_use = [int(1 + num_slices/4), int(1 + num_slices/2), int(1 + 3*num_slices/4)]
        elif self.numShortSlices == 5:
            short_slices_to_use = [int(1), int(1 + num_slices/4), int(1 + num_slices/2), int(1 + 3*num_slices/4), int(num_slices)]

        for slice in short_slices_to_use:
            slice_imgs_paths = list(filter(lambda x: '_slice_' + str(slice) + '_' in x, short_imgs_paths))
            slice_imgs = []
            for slice_img in slice_imgs_paths:
                img = cv2.imread(os.path.join(self.patient_dir, patient, slice_img))

                if self.transform is not None:
                    augmented = self.transform(image=img)
                    img = augmented['image']
                
                img = img.astype('float32') / 255
                img = img.transpose(2, 0, 1)
                img = np.expand_dims(img, axis=0)

                slice_imgs.append(img)
            
            short_imgs.append(slice_imgs)
        
        return (long_imgs, short_imgs), volumes, patient

# class GetMasksDataset(torch.utils.data.Dataset):
#     def __init__(self, img_ids, img_dir, mask_dir, img_ext, mask_ext, num_classes, transform=None):
#         """
#         Args:
#             img_ids (list): Image ids.
#             img_dir: Image file directory.
#             mask_dir: Mask file directory.
#             img_ext (str): Image file extension.
#             mask_ext (str): Mask file extension.
#             num_classes (int): Number of classes.
#             transform (Compose, optional): Compose transforms of albumentations. Defaults to None.
        
#         Note:
#             Make sure to put the files as the following structure:
#             <dataset name>
#             ├── images
#             |   ├── 0a7e06.png
#             │   ├── 0aab0a.png
#             │   ├── 0b1761.png
#             │   ├── ...
#             |
#             └── masks
#                 ├── 0
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 |
#                 ├── 1
#                 |   ├── 0a7e06.png
#                 |   ├── 0aab0a.png
#                 |   ├── 0b1761.png
#                 |   ├── ...
#                 ...
#         """
#         self.img_ids = img_ids
#         self.img_dir = img_dir
#         self.mask_dir = mask_dir
#         self.img_ext = img_ext
#         self.mask_ext = mask_ext
#         self.num_classes = num_classes
#         self.transform = transform

#     def __len__(self):
#         return len(self.img_ids)

#     def __getitem__(self, idx):
#         img_id = self.img_ids[idx]
        
#         img = cv2.imread(os.path.join(self.img_dir, img_id + self.img_ext))

#         mask = []
#         for i in range(self.num_classes):
#             mask.append(np.zeros(img.shape))
#         mask = np.dstack(mask)

#         if self.transform is not None:
#             augmented = self.transform(image=img, mask=mask)
#             img = augmented['image']
#             mask = augmented['mask']
        
#         img = img.astype('float32') / 255
#         img = img.transpose(2, 0, 1)
#         mask = mask.astype('float32') / 255
#         mask = mask.transpose(2, 0, 1)
        
#         return img, mask, {'img_id': img_id, 'patient': img_id[-2:]}