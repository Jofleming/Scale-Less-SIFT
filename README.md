# Scale-Less SIFT (SLS) OpenCV implementation
### University of Washington Bothell
### Authors: Jordan Fleming and James Woo

## Overview

This project is an implementation study of the Scale-Less SIFT (SLS) method introduced by Hassner et al.
The original paper showed that dense SIFT descriptors struggle when images contain large scale changes,
because most pixels do not have stable scale estimates. SLS addresses this problem by extracting descriptors
at multiple scales and compressing them into a scale-robust representation.

This repository contains a C++ implementation of the SLS pipeline written entirely using OpenCV.
The goal of the project is to reproduce the behavior described in the paper,
compare SLS to a dense SIFT (DSIFT) baseline, and evaluate differences in performance and match quality.

## What This Implementation Does

The program loads a pair of images called source.jpg and target.jpg.
Both images are downsampled to reduce computation time. The program then computes two types of descriptors:

Dense SIFT (DSIFT): One SIFT descriptor per pixel on a regular grid.
Scale-Less SIFT (SLS): Multi-scale SIFT descriptors combined using PCA and averaging,
approximating the scale-free representation from the paper.

After descriptors are computed, the program performs feature matching separately for DSIFT
and SLS using a FLANN nearest-neighbor search. 
The matches are drawn and saved as output images (matches_dsift.jpg and matches_sls.jpg),
and some performance statistics are printed to the console.

This allows direct comparison between DSIFT and SLS using the same image pair.

## How to Build

The project is designed to compile in Visual Studio with OpenCV installed. To build:

Open the Visual Studio solution.

Confirm that OpenCV include directories and library directories are correctly configured.

Build the project in Debug x64 or Release x64 mode.

The executable (SLS.exe) will be created in the appropriate build folder.

This project does not require any external libraries beyond OpenCV.

## How to Run

Place your input images in a folder named data inside the build directory. The files must be named:

source.jpg
target.jpg

Run the program from within Visual Studio or directly by running SLS.exe from the build directory.
The program will print progress information, compute DSIFT and SLS descriptors, 
match them, and save the visualization images in the same folder as the executable.

The final files produced are:

matches_dsift.jpg
matches_sls.jpg

These show the match correspondences for DSIFT and SLS.

## Results Summary

In our tests, both descriptor types ran successfully on the chosen image pair. 
DSIFT computed descriptors slightly faster, but SLS significantly reduced matching time because PCA 
reduced descriptor dimensionality. SLS also produced stronger and more stable matches when scale differences 
were present.

The performance results for our images were:

DSIFT extraction took about 5.2 seconds
SLS extraction took about 5.5 seconds
Matching was faster for SLS than DSIFT
SLS produced lower average match distances, showing improved match quality

The visual results also showed cleaner and more coherent matches for SLS.

## Important Notes About the Implementation

The original SLS algorithm requires constructing an upper-triangular matrix representation for subspace distances. 
This step is extremely expensive and impractical for real-time computation. 
The authors themselves provide a simplified “paper” mode in their MATLAB code.

This reproduction uses a streamlined approach:

Instead of computing the full Grassmann manifold mapping, PCA is used to reduce multi-scale descriptors, 
and the descriptors are averaged.

This follows the recommendation from the public SLS distribution and keeps runtime manageable while preserving 
the main idea.

## Included Features

Dense SIFT descriptor extraction
Multi-scale SIFT extraction for SLS
PCA dimensionality reduction
SLS descriptor construction
FLANN-based descriptor matching
Match visualization and output
Performance timing for extraction and matching

## Running Your Own Images

To test different image pairs, replace source.jpg and target.jpg in the data directory. 
Images with scale changes or object size differences will best illustrate the advantage of SLS.

## Reference

Hassner, T., Filosof, S., Mayzels, V., & Zelnik-Manor, L.
“SIFTing Through Scales.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017.
