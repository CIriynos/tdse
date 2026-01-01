cmake_minimum_required(VERSION 3.15)

project(CPPTDSE CXX)
string(TIMESTAMP COMPILE_TIME %Y%m%d-%H%M%S)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_compile_options(
    -Werror -Wunknown-pragmas
    -fopenmp
    -march=native
    -O3 -ffast-math
    # -g 
)

add_link_options(
    # -lfftw3_omp -lfftw3 -lm
    -fopenmp
    # -g
)

# add_definitions(-DEIGEN_DONT_PARALLELIZE)

set(SRC_LIST 
    ./src/interface.cpp 
    ./src/main.cpp
)

# set(INC_DIR /usr/local/include)
# set(LINK_DIR /usr/local/lib)

add_library(${PROJECT_NAME} ${SRC_LIST})

# include_directories(${INC_DIR})
# link_directories(${LINK_DIR})

target_link_libraries(
    ${PROJECT_NAME} PUBLIC 
    #fftw3
    #fftw3_omp
    OpenMP::OpenMP_CXX
)

target_include_directories(
    ${PROJECT_NAME} PUBLIC 
    ${PROJECT_BINARY_DIR} 
    ${PROJECT_SOURCE_DIR}
)
