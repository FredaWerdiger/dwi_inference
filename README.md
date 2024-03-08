# Public code to run inference on new DWI images using trained model
## Trained Models
Two trained models are available, one that uses the DWI and ADC images, and one that uses DWI alone if ADC is not available. 
## Input Data
organise_data.py brings data into the format necessary for running inference. However, it is based on the ATLAS dataset. If unable to use organise_data.py for lack of access to ATLAS, instructions for organising data are as follows:

- Images should be in NIFTI format (https://github.com/rordenlab/dcm2niix will convert from standard DICOM)
- DWI and ADC images should be stacked together as a 2-channel NIFTI (infer.py expects them to be stacked as [DWI, ADC])
- If there is no ADC, infer.py still expects a 2-channel images, so organise_data.py stacks the DWI twice as [DWI, DWI]
- The images should be organised in the same folder as follows: <path_to_images>/images/*subject*_image.nii.gz
- Predictions/segmentations will be there found as follows: <path_to_images>/pred/*subject*_image_pred.nii.gz
## Dependencies
MONAI version: 0.10.dev2237
Numpy version: 1.21.2
Pytorch version: 1.10.2
### Optional dependencies:
Pytorch Ignite version: 0.4.8
Nibabel version: 3.2.1
scikit-image version: 0.19.1
Pillow version: 8.4.0
TorchVision version: 0.11.3
tqdm version: 4.65.0
psutil version: 5.9.0
pandas version: 1.4.0

## How to cite this in the literature 
If used, please cite in literature as

Freda Werdiger, Vignan Yogendrakumar, Milanka Visser, James Kolacz, Christina Lam, Mitchell Hill, Chushuang Chen, Mark W. Parsons, Andrew Bivard,
Clinical performance review for 3-D Deep Learning segmentation of stroke infarct from diffusion-weighted images,
Neuroimage: Reports,
Volume 4, Issue 1,
2024,
100196, https://doi.org/10.1016/j.ynirp.2024.100196
