# video-cluster-analysis
Summarize the contents of a video using image clustering

## Motivation
Many applications involve collecting a continuous stream of signal data and detecting objects of interest. 
These signals may contain events of interest that are rare or not previously captured, which can make them
difficult to detect automatically. For example, a robot that autonomously navigates in an enviroment using
computer vision may be well designed to handle typical scenarios, but if a new environment or object not
previously seen is encountered that results in poor performance, it may be difficult to systematically
understand the issue without exhaustive manual analysis of the source video.

This project demonstrates an approach to this problem by performing an unsupervised analysis of a YouTube
video feed. By dividing each frame into smaller subimages and extracting relatively generalized image
features, trends and outliers can be detected in an automated and quantitative way to aid manual analysis
of the source video.

## Experiment
The following diagram outlines the high level steps of the `cluster_video.py` scripy:

![Experiment diagram](diagram.svg)

### Setup
Python 3.9.7 was used for this experiment. The dependencies for the project can be installed with `make install`,
preferably inside a virtual environment.

The script `download_yt_video.py` can also be used to download a source video from YouTube to use for analysis.

### Iterate through video frames
The script loops through the frames of a video whose path is provided with a command line argument.
Since typical YouTube videos run at about 30 frames per second and subsequent frames are likely to contain
similar information, the script skips about a second's worth of frames to reduce the amount of computation
required.

### Extract subframes
For each video frame, a number of "subframes" consisting of non-overlapping square image tiles are identified.
The metadata of each subframe including its source frame number, timestamp, and pixel ranges are stored for
future analysis.

### Extract features
Even small subframes consisting of 40 x 40 pixels and 3 color channels contain 4,800 dimensions, a relatively
high number that that may cause poor clustering performance due to
[the curse of dimensionality](https://en.wikipedia.org/wiki/Curse_of_dimensionality).

For each subframe, features consisting of a histogram of each color channel are extracted to reduce the
dimensionality of the subframes and allow for matching subframes with similar colors based on Euclidean
distance.

### Cluster features
The clustering step first consists of standardizing the extracted features to have zero mean and unit
variance.  This standardization is particularly useful with a nearest-neighbors classifier, since it ensures
each feature is given equivalent "weight" when measuring distance.

The standardized features are then clustered using
[K-nearest neighbors](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm). For identifying specific
objects in relatively complex source video, a setting of several hundred clusters may be appropriate.

## Results
TODO

### Discussion
Difficult problem due to a range of both illumination conditions and viewing angles

### Limitations
A K-nearest neighbors approach requires specifying a number of clusters. For unknown source material, an
effective number may be difficult to identify. A clustering algorithm such as
[mean shift](https://en.wikipedia.org/wiki/Mean_shift) may be more appropriate, which requires a 
cluster bandwidth parameter instead of a set number of clusters.
