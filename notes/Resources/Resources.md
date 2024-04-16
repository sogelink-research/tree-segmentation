# Resources

## Deep Learning methods

### Individual Tree Species Identification for Complex Coniferous and Broad-Leaved Mixed Forests Based on Deep Learning Combined with UAV LiDAR Data and RGB Images

See [here](https://www.mdpi.com/1999-4907/15/2/293). Really interesting paper which develops a variation of [YOLO v8](https://github.com/ultralytics/ultralytics) to combine LiDAR point clouds and RGB images. They also focused on topics really similar to what we are looking for: species identification and dense forests. Sadly, there is no implementation available online.

Some resources to help with the implementation:

- YOLOv8 [implementation](https://github.com/ultralytics/ultralytics), [structure](https://github.com/ultralytics/ultralytics/issues/189) and [documentation](https://docs.ultralytics.com/models/yolov8/)
- [Gather and Distribute](https://arxiv.org/pdf/2309.11331.pdf) (GAD) with an [implementation](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) in Pytorch
- [Convolutional Block Attention Module](https://github.com/Peachypie98/CBAM) (CBAM) in Pytorch for the AMFNet
- [ShuffleNet v2](https://github.com/pytorch/vision/blob/main/torchvision/models/shufflenetv2.py) in Pytorch for the AMFNet

### Tree segmentation from AHN4 Using a non-end-to-end neural network and random forest

See [here](https://repository.tudelft.nl/islandora/object/uuid%3A5d2ad31c-476e-4048-83f2-a5b4f92494d1).

### Tree detection from satellite images

See [here](https://github.com/talhayavcin/Tree-detection-from-satellite-images). This page is interesting because we have a [whole Jupyter notebook](https://github.com/talhayavcin/Tree-detection-from-satellite-images/blob/main/YOLOv5_Custom_Training.ipynb) explaining how they install YOLOv5 and create their own dataset from images.

### Forest 3D App

See [here](https://github.com/lloydwindrim/forest_3d_app)

Uses YOLOv3 and point clouds only. Seems flexible and could be used to train a first model.

### Automatic modelling of 3d trees using aerial lidar point cloud data and deep learning

See [here](https://ris.utwente.nl/ws/portalfiles/portal/276874011/Kippers2021automatic.pdf). Uses LiDAR point clouds processed in two steps:

1. Semantic segmentation using PointNet
2. Segmentation using the Watershed algorithm

### Tree Detection in Aerial Point Clouds

See [here](https://github.com/Amsterdam-AI-Team/Tree_Detection_in_Aerial_Point_Clouds). The method in itself doesn't seem so great but they have notebooks explaining how they preprocessed the data.

## Datasets

### Tools to label images

- [Labelimg](https://github.com/HumanSignal/labelImg) for bounding box labeling
- [SamGeo](https://samgeo.gishub.org/examples/box_prompts/) to segment images using SAM and bounding boxes

### Satellite images

IGN data [here](https://geoservices.ign.fr/bdortho)

### MillionTrees

See [here](https://milliontrees.idtrees.org/). See if there is anything to get from this.

### French data

- French government data page: [here](https://www.data.gouv.fr/fr/datasets/?page=2&q=arbre)
- Bordeaux: [here](http://www.opendata.bordeaux.fr/content/patrimoine-arbore) or better [here](https://opendata.bordeaux-metropole.fr/explore/dataset/ec_arbre_p/information/?disjunctive.insee)
- Grenoble: [here](https://data.metropolegrenoble.fr/visualisation/information/?id=arbres-grenoble)
- Lyon: [alignment trees](https://data.grandlyon.com/portail/fr/jeux-de-donnees/arbres-alignement-metropole-lyon/donnees)
- Montpellier: [without position](http://data.montpellier3m.fr/dataset/arbres-dalignement-de-montpellier)
- Mulhouse: [here](https://data.mulhouse-alsace.fr/explore/dataset/68224_arbres_alignement/information/)
- Toulouse: [here](https://data.toulouse-metropole.fr/explore/dataset/arbres-urbains/information/)
- Nice: [here](https://opendata.nicecotedazur.org/data/dataset/cartographie-des-arbres-communaux)

### Dutch data

#### AHN, Lidar Pointclouds countrywide

Several versions with 5 years in between, most recent one is from last year (AHN4).
All point clouds have been shot during winter time so no leafs, good for finding internal tree structure and ground.
Data can be downloaded in (very) large chunks from [here](https://ahn.arcgisonline.nl/ahnviewer/).
This raw data needs to be processed into smaller subsets first before something can be done with it (see PDAL).
These large chunks have been divided into 25 chunks each and can be found [here](https://geotiles.citg.tudelft.nl/).

#### Luchtfotos (Aerial pictures)

Aerial pictures of every year. High res (~8cm) in RGB and lower res (25cm) in RGB and IR.
Data can be downloaded from [here](https://www.beeldmateriaal.nl/data-room).

#### Trees Amsterdam

All 300.000 trees managed by the municipality of Amsterdam, including species, height, diameter, age and xy location (but excluding crown shape).
Can be downloaded [here](https://maps.amsterdam.nl/open_geodata/?k=505).

### Other sources

- [OpenTrees.org](https://opentrees.org/)
- Some potentially interesting data from mountains [here](https://esajournals.onlinelibrary.wiley.com/doi/10.1002/ecy.1759)

## Software

- [QGIS](https://qgis.org/en/site/), generic GIS viewer, good for checking any set of geographic data
- [GDAL](https://gdal.org/), good for handling any type of vector or raster geo dataset (both CLI and python)
- [PDAL](https://pdal.io/), good for handling point cloud datasets, can be used for creating small subsets out of larger ones
- [Cloud Compare](https://www.danielgm.net/cc/), powerful viewer for point clouds
