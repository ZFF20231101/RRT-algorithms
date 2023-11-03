# RRT-algorithms
Path plannning: RRT* and other algorithms.

# Usage
Each file can be run directly. The file simulating 2D space contains 4 maps and 3D space contains 2 maps. Maps, start and target point locations, maximum number of iterations can be selected prior to running. Running it will bring up a pop-up window with the plotting results. Each iteration the random tree will expand outwards once, adding a node and a path. The algorithm will stop after the maximum number of iterations and plot the final path as well as print the path and time data.

# Requirements
The following packages need to be added before running the file.
- [Python 3+](https://www.python.org/downloads/)
- [matplotlib](https://matplotlib.org/)
- [numpy](https://numpy.org/)
- [Shapely](https://pypi.org/project/shapely/)
