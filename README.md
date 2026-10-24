# CS477 Assignment 2 Harris Corner Detection

Python 3.12.14 was used with NumPy 2.4.2 and opencv-python-headless 4.13.0.92 (cv2 4.13.0).

Supporting files for the CS477 Harris Corner Detection assignment. The PDF report is submitted separately. This repository contains the complete program, README, pinned dependencies, test images, experiment configuration, raw response maps, overlays, CSV and JSON results, and verification files. The function/parameter/output table is included as function_parameter_output_table.csv and also appears in Part C of the report.

Run from this directory:

```
python -m pip install -r requirements.txt
python experiment.py --config config.json --output results
```

The entry point is configurable through --config and --output. The supplied synthetic geometry uses size 384. Edit the baseline or sweep values in config.json as needed; keep odd supports and a border large enough for derivative, integration and NMS support. The supplied border is 6 pixels, adequate for all required settings. Seed 47702 generates Gaussian noise with standard deviation 0.02, followed by clipping to [0,1].

Each of 36 unique runs has response.npy (384 by 384 float32), threshold_mask.png (threshold before NMS and border removal), corners.npy and corners.csv (K by 2, x,y, zero-based), and overlay.png. Natural image metrics are intentionally absent because there is no annotated ground truth. Full precision values and median runtimes are in results/results.csv. The required baseline rows also supply the repeated baseline entries in each parameter sweep.

The common NMS uses 3 by 3 dilation, exact equal-response 8-connected plateau components, and the lowest row then column for ties. No distance suppression or corner count cap is applied. Reflect-101 filtering is followed by common six-pixel border exclusion. Thresholds use the full raw response maximum separately for each detector. Non-positive maxima produce no corners. DT is zero when the eigenvalue sum is at most 1e-12.

Matching maximizes one-to-one matches within 3 Euclidean pixels through deterministic augmenting paths. Error is averaged over matched pairs; None/blank means not applicable. The known corners are well separated. Timing includes response computation and final point selection, one warm-up then median of five runs, one CPU thread and OpenCL off. It excludes I/O, image generation, evaluation and plotting. Each branch recomputes its own response, including gradients and tensor.

The original photographs are included so no network is needed to reproduce the results. Images are converted from BGR to grayscale, center-cropped to square and resized to 384 by 384 with INTER_AREA. The synthetic and noisy .npy images retain float32 values; PNGs are rounded display copies.

Image sources:

- https://github.com/opencv/opencv/blob/4.13.0/samples/data/building.jpg
- https://github.com/opencv/opencv/blob/4.13.0/samples/data/fruits.jpg

API reference: https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html

External code: OpenCV and NumPy are dependencies, not code written by the student. No third-party listing was copied. Python standard-library modules handle file formats, timing and arguments. The original photographs come from the cited OpenCV repository.

## Repeatability check

Keep the original results folder and run:

```
python experiment.py --config config.json --output reproduced_results
python verify_reproduction.py --reference results --candidate reproduced_results
```

A fresh run in the recorded environment matched all 36 response maps bit for bit, all corner coordinate arrays exactly and all non-timing fields. Runtime values can vary. The test checks exact agreement; another software build or machine can introduce floating-point differences. The original submitted timings have not been replaced.

## Sources

[1] OpenCV. Feature Detection API, version 4.13.0. cornerMinEigenVal, cornerHarris and cornerEigenValsAndVecs. https://docs.opencv.org/4.13.0/dd/d1a/group__imgproc__feature.html

[2] OpenCV. building.jpg, sample photograph, release 4.13.0. https://github.com/opencv/opencv/blob/4.13.0/samples/data/building.jpg

[3] OpenCV. fruits.jpg, sample photograph, release 4.13.0. https://github.com/opencv/opencv/blob/4.13.0/samples/data/fruits.jpg

[4] Harris, C., and Stephens, M. (1988). A Combined Corner and Edge Detector. Alvey Vision Conference. DOI 10.5244/C.2.23. https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.html

[5] Shi, J., and Tomasi, C. (1994). Good Features to Track. IEEE CVPR, pp. 593-600. DOI 10.1109/CVPR.1994.323794. https://scholars.duke.edu/publication/802423

[6] Canny, J. (1986). A Computational Approach to Edge Detection. IEEE TPAMI, 8(6), 679-698. DOI 10.1109/TPAMI.1986.4767851. https://www.cs.princeton.edu/courses/archive/fall13/cos429/papers/Canny86.pdf

[7] OpenCV. Image Filtering API, version 4.13.0. Sobel, filter2D, Gaussian kernels and dilation. https://docs.opencv.org/4.13.0/d4/d86/group__imgproc__filter.html

[8] NumPy. Generator.normal, NumPy 2.4 documentation. https://numpy.org/doc/2.4/reference/random/generated/numpy.random.Generator.normal.html

[10] SEECS. CS477 Assignment 2 Harris Corner Detection. Course assignment handout supplied for this submission, pp. 1-6.
