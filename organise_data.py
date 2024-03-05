#! /usr/bin/env python3
import os
import sys
import csv
import pandas as pd
from nipype.interfaces.image import Reorient
from nipype.interfaces.fsl.preprocess import BET
from nibabel.processing import resample_from_to
import nibabel as nb
import time
import glob

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
The location of the mediaflux drive (e.g. Y:/)
out_path:
the location of the result. Within the specified path, the stacked images will be saved under 'images'
'''


def main(subjects_file, atlas_path, out_path, overwrite=False):
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    if not os.path.exists(os.path.join(out_path, 'images')):
        os.makedirs(os.path.join(out_path, 'images'))
    atlas = os.path.join(atlas_path, 'ATLAS_database')
    subjects_df = pd.read_csv(subjects_file, sep=',', header=None, names=['subject'])
    subjects_df['adc'] = ''
    subjects_df['dwi'] = ''
    subjects = subjects_df.subject.to_list()
    dwi_paths = []
    adc_paths = []
    for name in subjects:
        print(name)
        DWI_loc = os.path.join(atlas, name, 'MR-follow_up/DWI-follow_up', name + '-fu-DWI_b1000.nii.gz')
        ADC_loc = os.path.join(atlas, name, 'MR-follow_up/ADC-follow_up', name + '-fu-ADC.nii.gz')
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
    bet_fault = []
    orientation = []
    # initiate BET
    betdwi = BET()
    betdwi.inputs.frac = 0.3
    betadc = BET()
    betadc.inputs.frac = 0.1

    # temporary image for the BET outputs
    temp1 = 'temp1.nii.gz'

    subjects = subjects_df.subject.to_list()
    for subject in subjects[20:]:
        if not overwrite:
            if os.path.exists(os.path.join(out_path, 'images', subject + '_image.nii.gz')):
                continue
        print(f"Running for subject: {subject}")
        try:
            dwi = [file for file in dwi_paths if subject in file][0]
        except IndexError:
            continue

        # run bet on ADC (0.1) and DWI (0.3)
        betdwi.inputs.in_file = dwi
        betdwi.inputs.out_file = temp1
        # create mask
        betdwi.inputs.mask = True
        try:
            betdwi.run()
        except RuntimeError:
            bet_fault.append(subject)
            continue
        time.sleep(3)
        dwi_im = nb.load(temp1)
        mask = temp1.split('.nii')[0] + '_mask.nii.gz'
        mask_im = nb.load(mask)

        try:
            adc = [file for file in adc_paths if subject in file][0]
        except IndexError:
            print('No ADC')
            continue

        adc_im = nb.load(adc)

        if len(adc_im.get_fdata().shape) == 4:
            adc_check.append(adc)
            adc_im = nb.Nifti1Image(adc_im.get_fdata()[:, :, :, 0],
                                    header=adc_im.header,
                                    affine=adc_im.affine)
        if len(adc_im.get_fdata().shape) > 4:
            adc_check.append(subject)
            continue

        if not dwi_im.get_fdata().shape == adc_im.get_fdata().shape:

            print('ADC not the same size as DWI. Resizing...')
            adc_check.append(adc)
            # resize adc im to be the same size as the dwi image
            adc_im = resample_from_to(adc_im, dwi_im)

        # Check orientations of adc and dwi
        dwi_or = nb.aff2axcodes(dwi_im.affine)
        adc_or = nb.aff2axcodes(adc_im.affine)

        if not adc_or == dwi_or:
            print('ADC and DWI are not orientated the same direction')
            orientation.append(subject)
            reorient_to = dwi_or[0] + dwi_or[1] + dwi_or[2]
            print('Reorienting ADC to the same orientation as DWI')
            reorient = Reorient(orientation=reorient_to)

            reorient.inputs.in_file = adc
            res = reorient.run()
            time.sleep(3)
            adc_im = nb.load(res.outputs.out_file)


        # mask adc
        adc_masked = adc_im.get_fdata() * mask_im.get_fdata()
        adc_masked_im = nb.Nifti1Image(adc_masked, header=adc_im.header, affine=adc_im.affine)

        combined_im = nb.concat_images([dwi_im, adc_masked_im], check_affines=False)
        nb.save(combined_im,
                os.path.join(out_path, 'images', subject + '_image.nii.gz')
                )

        # remove reorientation file
        if not adc_or == dwi_or:
            os.remove(res.outputs.out_file)
            mat_file = glob.glob('*.mat')
            [os.remove(file) for file in mat_file]

    with open('bet_fault.csv', 'w') as myfile:
        writer = csv.writer(myfile)
        for subject in bet_fault:
            writer.writerow([subject])
    with open('check_adc.csv', 'w') as myfile:
        writer = csv.writer(myfile)
        for subject in adc_check:
            writer.writerow([subject])
        for subject in orientation:
            writer.writerow([subject])


if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Missing arguments.')
        print('Usage: python organise_data.py subjects_csv atlas_path out_path optional:overwrite=True/False')
    else:
        main(*sys.argv[1:])
