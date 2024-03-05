import csv
import glob
import os


HOMEDIR = os.path.expanduser('~/')

atlas = HOMEDIR + 'atlas'

patients = os.listdir(atlas + '/ATLAS_database')

adcs = glob.glob(atlas + '/ATLAS_database/*/MR-follow_up/ADC-follow_up*/*.nii.gz')
dwis = glob.glob(atlas + '/ATLAS_database/*/MR-follow_up/DWI-follow_up*/*b1000.nii.gz')
lesions = glob.glob(atlas + '/ATLAS_database/*/MR-follow_up/DWI-follow_up*/*lesion.nii.gz')


patients_dwi = [patient for patient in patients if
                (any(patient in path for path in adcs) and
                 any(patient in path for path in dwis) and not
                any(patient in path for path in lesions))
]

patients_dwi.sort()

with open('all_atlas.csv', 'w') as myfile:
    writer = csv.writer(myfile)
    for subject in patients_dwi:
        writer.writerow([subject])