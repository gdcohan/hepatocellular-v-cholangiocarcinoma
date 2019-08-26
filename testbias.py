#!/usr/bin/python
import SimpleITK as sitk
import os
from glob import glob
import nibabel as nib
import numpy as np

def n4_bias_correction(image):
    import ants
    as_ants = ants.from_numpy(image)
    corrected = ants.n4_bias_field_correction(as_ants)
    return corrected.numpy()


def stkbias(image, image_type=sitk.sitkFloat64):
    input_image = sitk.ReadImage(image, image_type)
    output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
    return output_image


patho = '/media/user1/my4TB/robin/ovarian/ovarian_data/test'
baseDir = os.path.normpath(patho)
files = glob(baseDir + '/*/T1POST/*.nii')

for file in files:
    filePath, fileName = os.path.split(file)
    a = filePath.split('\\')


    startPath = '/media/user1/my4TB/robin/ovarian/ovarian_data/normalized'
    nePath = startPath + a[5]
    neePath = nePath + '/T1POST'
    newPath = neePath + '/' + fileName

    if not os.path.isdir(startPath):
        os.mkdir(startPath)

    if not os.path.isdir(nePath):
        os.mkdir(nePath)

    if not os.path.isdir(neePath):
        os.mkdir(neePath)

    n1_img = nib.load(file)
    n1_header = n1_img.header
    n1_affine = n1_img.affine
    img = n1_img.get_fdata()
    img = n4_bias_correction(img)
    imgmax = img.max()
    img = np.true_divide(img, imgmax)
    new_img = nib.Nifti1Image(img, n1_affine, n1_header)
    nib.save(new_img, newPath)