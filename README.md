# # Codes for paper-  OPHash: Learning of Organ and Pathology Context Sensitive Hashing for Medical Image Retrieval


## Prerequisites
* Ubuntu\* 20.04
* Python\* 3.9
* NVidia\* GPU for training
* 16GB RAM for inference
* CUDA 11.2


## Datasets
### Large size dataset:
  Brain [(Figshare)](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427/5), Breast [(link)](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) and Chest and Retina [(link)](https://data.mendeley.com/datasets/rscbjbr9sj/2).
### Small size dataset:
MedMNIST v2. (#medmnist) [(link)](https://zenodo.org/record/6496656), [(link2)](https://medmnist.com/)


Sample dataset availabel on './data/'.

## Train
```
python OPHash.py
```
Trained model will be saved in './models/'.
Trained models are available in [Here](https://iitkgpacin-my.sharepoint.com/:f:/g/personal/asimmanna17_kgpian_iitkgp_ac_in/EjehZrRXBR1BvVr6LGArxv8BiDBI3Wgk_Il_qeZJLCthwQ?e=TtqSLD).

## Evalutions

```
python evalution.py
```


**Contributor**

The codes/model is contributed  by

<a href="https://www.linkedin.com/in/asimmanna17/">Asim Manna</a>, </br>
Centre of Excellence in Artificial Intelligence, </br>
Indian Institute of Technology Kharagpur </br>
email: asimmanna17@kgpian.iitkgp.ac.in </br> 

