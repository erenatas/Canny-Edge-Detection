# Canny Edge Detection

## Running Instructions

### Required Tools
Python 3.7 - [Miniconda](https://conda.io/miniconda.html)

### Required Libraries to Install
```conda install scipy matplotlib numpy```

### Running arguments 

**There are 3 arguments that python file takes:**
  —-image : Path of the image file
  —-sigma : standard deviation of sigma
  —-minthreshold : Minimum Threshold for Double Thresholding
  —-maxthreshold : Maximum Threshold for Double Thresholding

**Note**: You can also use ```python CannyEdgeDetection.p```y -h for help with arguments.

And the default values for these arguments are:
```
image: im01.jpg
Sigma: 1.4
minthreshold: 100
maxthreshold: 200
```
You can simply run by writing:
```python CannyEdgeDetection.py --image im02.jpg --sigma 1.4 --minthreshold 100 --maxthreshold 200```

Or
```python CannyEdgeDetection.py```
 
For using it with default values.