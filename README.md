# Arundo Mapper

This project attempts to map Arundo donax (an invasive giant reed) along the Santa Clara River in Southern California. Input datasets include:

1. Ground Truth Maps
- produced by Stillwater Sciences with aerial imagery, lidar, and field surveys in 2018
- document Arundo cover in classes ranging from 1 to 95% cover
- maps vegetation types and associations

2. Remote Sensing Datasets
- lidar - collected in 2015, 2018, and 2024
- satellite phenology spectral mixture analysis results 
- mid-summer spectral reflectance 
- aerial imagery from the National Agricultural Imagery Program (NAIP) - RGB+NIR at 0.6 m

The main script here loads the ground truth maps and remote sensing data, then applies a Random Forest (with the Ranger package in R) to map out Arundo class types from the satellite imagery.

We implement two test versions. First, a version predicting the actual Arundo class (e.g. 0-1%, 1-5%, ..., 75-95%, 95-100%). Next, a version which produces binary maps thresholded at each class level (e.g. 0-1%, 0-5%, 0-10%, ..., 0-95%, 95-100%). 

In all classifiers, we use training-validation-test splits of 70-15-15. Each individual polygon is sampled repeatedly within each Landsat pixel scale (30 m), such that one large polygon might supply many different individual samples, but each polygon should be allowed to occur ONLY within the training, validation, or test sets (not leaking across them). 

After training the random forest, the following outputs are made for both sets of classifiers;
- confusion matrices
- estimates of overall, user's, producer's, and balanced accuracy for each class (and overall), plus the Kappa statistic
- actual maps of Arundo cover