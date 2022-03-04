# FaceAdv

It is FaceAdv, a physical-world attack that crafts adversarial stickers to deceive Face Recognition (FR) systems.

## How to work

1. ```preprocess/detect_landmarks_in_image.py``` to clip images and detect landmarks.
2. ```preprocess/recon.py``` to obtain parameters of 3D face reconstruction.
3. ```faceadv/face_adv.py``` to train generator model to generate adversarial stickers.

## File Structure

```
|- Auxiliary                    Saving auxiliary files
|- FaceAdv
|-------|
        |- module                Target FR systems
        |- config.txt            Configuration files
        |- face_adv.py           Main file
        |- train.ps1/test.ps1    Training scripts / Testing scripts
|- Preprocess
|-------|
        |- BFM                  Configuration files
        |- dataset              Clipped images
        |- facebank             Captured images
        |- output               3D Face Reconstrcution parameters
        |- models               Model architecture and model parameters
|- Video                        Face recognition systems for evaluating physical attacks
```


## Reference
```
@article{shen2021effective,
  author    = {Meng Shen and
               Hao Yu and
               Liehuang Zhu and
               Ke Xu and
               Qi Li and
               Jiankun Hu},
  title     = {Effective and Robust Physical-World Attacks on Deep Learning Face
               Recognition Systems},
  journal   = {{IEEE} Trans. Inf. Forensics Secur.},
  volume    = {16},
  pages     = {4063--4077},
  year      = {2021},
  url       = {https://doi.org/10.1109/TIFS.2021.3102492},
  doi       = {10.1109/TIFS.2021.3102492},
  timestamp = {Wed, 01 Sep 2021 12:46:09 +0200},
  biburl    = {https://dblp.org/rec/journals/tifs/ShenYZXLH21.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```