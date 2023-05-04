# LSMA Freigburg Forest Project
Final project for subject the MUTSC subject LSMA where [Freiburg Forest Dataset](http://deepscene.cs.uni-freiburg.de/segment_random/CITYSCAPES_RGB?da
taset=freiburgforest&model=FOREST_RGB) will be used.

## Description
Our final project aims to perform image segmentation on the Freiburg Forest dataset,
which comprises five different classes related to nature and diverse shapes
representing distinct spectral bandwidths. Two different methods will be explored to
achieve the objective. The first method will consist of having different image
descriptors for each class and classifying each pixel in the image so it will belong to
the class with the highest score obtained from the corresponding image descriptor.
Then, a more efficient approach will be also explored by implementing some simple
neural networks for classifying each of the pixels. These networks will use the most
relevant information determined by some data analysis that will be conducted on the
images to identify those bands that are more relevant. The goal then is to obtain an
image segmentation system based on different approaches, that can be also
extended to similar tasks.

## Project Planification
- Week 1: exploratory data analysis to identify the amount of samples that we
have for each class on the dataset and also to determine which are the most
relevant bands for the image segmentation step.
- Week 2: classification using image descriptors and comparison with the
actual segmentation provided with the dataset.
- Week 3: test first neural networks approaches and have some results to
compare with the actual image segmentation provided.
- Week 4: improvements over both methods and fine-tuning parameters to
obtain best possible models.
