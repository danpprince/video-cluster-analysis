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

![Experiment diagram](diagram.svg)


### Setup
Python 3.9.7 was used for this experiment. The dependencies for the project can be installed with `make install`,
preferably inside a virtual environment.

## Results
TODO

### Discussion
Difficult problem due to a range of both illumination conditions and viewing angles

