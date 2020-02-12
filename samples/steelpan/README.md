Installation instructions:


Clone the `Mask_RCNN` repo. 
Then 
`cd Mask_RCNN`
Put all this "steelpan" stuff in a new directory `Mask_RCNN/samples/steelpan/`

Note that python3.4 is required by package but not available via `conda`, so we use 3.5
```
conda create --name mrcnn  python=3.5     
conda activate mrcnn
cd Mask_RCNN/samples/steelpan/
```
and they say it's bad to mix conda a pip, but I couldn't manage to install the packages via conda ('Solving environment' hangs), so...
```
pip install -r requirements.txt
```

<!-- 
#conda install -f -y -q --name mrcnn -c anaconda --file requirements.txt
#conda install -f -y -q --name mrcnn -c conda-forge imgaug 
-->

To train: 
<!-- First get the COCO weights by running the first part of  `../demo.ipynb`.  This will put them in `../../mask_rcnn_coco.h5`. -->

Download a dataset of steelpan .png and .csv files and put them in `../../datasets/steelpan`, and have them already split up into 
`train/` and `val/` directories.  

Then run

    $ ./steelpan.py --weights imagenet --dataset ../../datasets/steelpan train

...more later
