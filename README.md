# Build instructions for Ubuntu 20.04

## Requirements

### Compiler supporting C++20
You need the full gcc toolchain supporting C++20 (e.g. g++-10):
 ```
 sudo apt install g++-10 gcc+-10
 ```

### Boost and OpenCV devel libraries
 ```
 sudo apt install libboost-dev libboost-program-options-dev libopencv-dev
 ```

### Build
 ```
 export CC=/usr/bin/gcc-10
 export CXX=/usr/bin/g++-10
 mkdir build
 cd build
 cmake /path/to/source/stereo_calib # replace the path with proper one
 cmake --build .
 ```

## Run the program with --help to see the command line options
 ```
 ./stereo_calib --help
 ```
