# Connecting rods inspection

## Description
This university project is an implementation of a visual system which is able to analyze
motorcycle connecting rods, by the following steps:
1. A grayscale image is loaded and binarized using `Otsu's method`
2. The image is bitwise inverted to get a white foreground and a black background
3. Morphological operations are carried out in order to detach touching rods
4. Blobs are extracted using `connected components labeling`
5. Blobs are filtered using a threshold on the blob's area, to remove iron powder and distractors
6. Circles are calculated using a custom `contour finding` method and the `Haralick's circularity` measure
7. Blob's `moments` are calculated using a custom method
8. Blob's `orientation` and its bounding box are determined
9. Blob's `shape features`, like length and width, are calculated
10. The number of holes inside each rod is computed, based on the `Euler number`
11. Finally, results are printed out

In the whole execution, images of the happening processing are shown for debug purposes.

## Dependencies
This software is written in `Python 3.7.6`, using the following third-party libraries:
* `plac 1.1.3`, to parse from CLI "the easy way"
* `scipy 1.4.1`, to efficiently perform scientific computations
* `opencv-python 4.2.0.32`, to exploit some computer vision algorithms

## Installation & execution
To install the software, just clone this repository locally.\
To execute it, `cd` into the downloaded folder (`cv-con-rod-inspection`) and run
```bash
python inspection.py -i "<image_path>"
```

## Todo
The last thing to refine is how to `detach touching rods`, without altering their main structure.
