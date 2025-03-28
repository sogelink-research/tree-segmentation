# Data Preprocessing

## Image size

- The YOLOv8 model takes as input images of size 640 $\times$ 640
- The size of the chunks of RGB images is 1 km $\times$ 1 km and their resolution is 8 cm, which amounts to 12 500 px per image.
- By trying different resolutions for creating CHM images from LiDAR point clouds, I found that 25 cm seems to be a great compromise. Therefore, aiming for 24 cm to be consistent with the 8 cm from the RGB images seems great

Then, we should extract images of 1920 $\times$ 1920 pixels to have a size 640 $\times$ 640 after a reduction to a 24 cm resolution. To have a proper overlap of images to have every tree in one piece in at least one of the images, we should aim for an overlap of about 40 m, which is quite large since images will be 153.6 m large. In terms of pixels, we can take 160 pixels as the margin (38.4 m).

With these parameters, we will get about 50 images per km², so 50 images from one single full RGB image. However, there will be overlaps between the different full images, which will sometimes require to use 4 different full (1 km²) images to compute one data image.

## Alignment of images with the point cloud

Due to the images not being orthophotos, it might be interesting to shift the images compared to the point cloud to align everything.
