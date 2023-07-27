# 20fold_fMRI

Deep Learning Reconstruction Enables 20-Fold Acceleration for 7T Whole-Brain fMRI

This is an implementation of "Deep Learning Reconstruction Enables 20-Fold Acceleration for 7T Whole-Brain fMRI"

© 2023 Regents of the University of Minnesota

"Deep Learning Reconstruction Enables 20-Fold Acceleration for 7T Whole-Brain fMRI" is copyrighted by Regents of the University of Minnesota. Regents of the University of Minnesota will license the use of "Deep Learning Reconstruction Enables 20-Fold Acceleration for 7T Whole-Brain fMRI" solely for educational and research purposes by non-profit institutions and US government agencies only. For other proposed uses, contact umotc@umn.edu. The software may not be sold or redistributed without prior approval. One may make copies of the software for their use provided that the copies, are not sold or distributed, are used under the same terms and conditions. As unestablished research software, this code is provided on an "as is'' basis without warranty of any kind, either expressed or implied. The downloading, or executing any part of this software constitutes an implicit agreement to these terms. These terms and conditions are subject to change at any time without prior notice.

Please cite the following:

Demirel, Omer Burak, et al. "20-fold accelerated 7T fMRI using referenceless self-supervised deep learning reconstruction." 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC). IEEE, 2021.

Demirel, Omer Burak, et al. "Improved simultaneous multi-slice functional MRI using self-supervised deep learning." 2021 55th Asilomar Conference on Signals, Systems, and Computers. IEEE, 2021.

This implementation is for 20-fold acceleration (5-fold SMS and 4-fold in-plane acceleration fMRI at 7T) with physics-guided self-supervised deep learning reconstruction. To train, please use main.py and the model that has been used in the paper can be found under savedModels folder. 

Here is the data structure:
RO:          # of readout lines,
PE:          # of phase encode lines,
No_C:        # of coil elements,
Slices:      # of slices,
Time-frames: # of time-frames phases,

Input data:
- kspace (RO x PE x NO_C x Dynamics)
- sense_maps (RO x PE x NO_C x Slices) with CAIPI shifts

![Output](images/res3.png)

 

