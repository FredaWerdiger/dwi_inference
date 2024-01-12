#! /usr/bin/env python3
import os
import sys

import pandas as pd
from nipype.interfaces.image import Reorient
from nipype.interfaces.fsl.preprocess import BET
from nibabel.processing import resample_from_to
import nibabel as nb
import time

'''
Arguments:
subject_file:
subject file input is expected to be comma separated values file of INSPIRE subject names
created like
with open('test_file.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    for subject in subs:
        writer.writerow([subject])
atlas_path:
The location of the mediaflux drive (e.g. Y:)
out_path:
the location of the result. Within the specified path, the stacked images will be saved under 'images'
'''

def main(subjects_file, atlas_path, out_path):
    atlas = os.path.join(atlas_path, 'ATLAS_database')
    subjects_df = pd.read_csv(subjects_file, sep=',', header=None, names=['subject'])
    subjects_df['adc'] = ''
    subjects_df['dwi'] = ''
    subjects = subjects_df.subject.to_list()
    dwi_paths = []
    adc_paths = []
    for name in subjects:
        print(name)
        DWI_loc = atlas + '/' + name + '/MR-follow_up/DWI-follow_up/' + name + '-fu-DWI_b1000.nii.gz'
        ADC_loc = atlas + '/' + name + '/MR-follow_up/ADC-follow_up/' + name + '-fu-ADC.nii.gz'
        if os.path.exists(DWI_loc):
            print("The DWI exists for {}. Adding path.".format(name))
            dwi_paths.append(DWI_loc)
            subjects_df.loc[subjects_df.subject == name, 'dwi'] = 1
            if os.path.exists(ADC_loc):
                print("The ADC exists for {}. Adding path.".format(name))
                adc_paths.append(ADC_loc)
                subjects_df.loc[subjects_df.subject == name, 'adc'] = 1
            else:
                subjects_df.loc[subjects_df.subject == name, 'adc'] = 0
        else:
            print("Patient {} has no follow-up DWI.".format(name))
            subjects_df.loc[subjects_df.subject == name, 'dwi'] = 0

    adc_check = []
    # initiate BET
    betdwi = BET()
    betdwi.inputs.frac = 0.5
    betadc = BET()
    betadc.inputs.frac = 0.1
    # initiate Reorient
    reorient = Reorient(orientation='RAS')

    # temporary image for the BET outputs
    temp1 = 'temp1.nii.gz'

    subjects_full_imaging = subjects_df[(subjects_df.dwi == 1) & (subjects_df.adc == 1)].subject.to_list()
    for subject in subjects_full_imaging:
        print(f"Running for subject: {subject}")
        dwi = [file for file in dwi_paths if subject in file][0]
        adc = [file for file in adc_paths if subject in file][0]

        # run bet on ADC (0.1) and DWI (0.5)
        betdwi.inputs.in_file = dwi
        betdwi.inputs.out_file = temp1
        # create mask
        betdwi.inputs.mask = True
        betdwi.run()
        time.sleep(3)
        mask = temp1.split('.nii')[0] + '_mask.nii.gz'

        reorient.inputs.in_file = temp1
        res = reorient.run()
        time.sleep(3)
        dwi_im = nb.load(res.outputs.out_file)

        reorient.inputs.in_file = mask
        res = reorient.run()
        time.sleep(3)
        mask_im = nb.load(res.outputs.out_file)

        reorient.inputs.in_file = adc
        res = reorient.run()
        time.sleep(3)
        adc_im = nb.load(res.outputs.out_file)
        if not dwi_im.get_fdata().shape == adc_im.get_fdata().shape:
            print('ADC not the same size as DWI. Resizing...')
            adc_check.append(adc)
            # resize adc im to be the same size as the dwi image
            adc_im = resample_from_to(adc_im, dwi_im)

        # mask adc
        adc_masked = adc_im.get_fdata() * mask_im.get_fdata()
        adc_masked_im = nb.Nifti1Image(adc_masked, header=adc_im.header, affine=adc_im.affine)
        # remove reoriented file
        os.remove(res.outputs.out_file)
        combined_im = nb.concat_images([dwi_im, adc_masked_im], check_affines=False)

        if not os.path.exists(os.path.join(out_path, 'images')):
            os.makedirs(os.path.join(out_path, 'images'))

        nb.save(combined_im,
                os.path.join(out_path, 'images', subject + '_image.nii.gz')
                )


if __name__ == '__main__':
    subjects_file = sys.argv[1]
    atlas_path = sys.argv[2]
    out_path = sys.argv[3]
    main(subjects_file, atlas_path, out_path)