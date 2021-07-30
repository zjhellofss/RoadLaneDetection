cmake_minimum_required(VERSION 3.19)
project(RoadLaneDetection)

set(CMAKE_CXX_STANDARD 14)

add_executable(RoadLaneDetection main.cpp source/kdtree.cpp source/NFA.cpp include/NFA.h include/dbscan.h include/kdtree.h include/image_utils.h source/image_utils.cpp include/lines.h)


#########################################################
# 全局宏定义
#########################################################
# 通用日志等级
add_compile_definitions(LDETAIL=1)
add_compile_definitions(LVERBOSE=2)
add_compile_definitions(LTRACE=3)

# 编译日期
string(TIMESTAMP BUILDDATE "%Y%m%d%H")
string(SUBSTRING ${BUILDDATE} 2 8 BUILDDATE)

# 版本号
set(VERSION_MAJOR 1)
set(VERSION_MINOR 0)
set(VERSION_PATCH 0)
set(VERSION_TWEAK ${BUILDDATE})
add_compile_definitions(VERSION_MAJOR=${VERSION_MAJOR})
add_compile_definitions(VERSION_MINOR=${VERSION_MINOR})
add_compile_definitions(VERSION_PATCH=${VERSION_PATCH})
add_compile_definitions(VERSION_TWEAK=${VERSION_TWEAK})

set(CMAKE_VERBOSE_MAKEFILE OFF)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_C_VISIBILITY_PRESET hidden)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)
set(CMAKE_SKIP_BUILD_RPATH FALSE)
add_compile_options(-fPIC -Wall)
add_compile_options(-msse2 -msse3 -msse4 -msse4.1 -msse4.2 -mssse3 -mavx -mavx2 -mfma -mpclmul)

################################################################
# 3rd libraries
################################################################

# boost
FIND_PACKAGE(Boost 1.76 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

# json
find_package(simdjson REQUIRED)
link_libraries(simdjson::simdjson)
message(STATUS "found lib simdjson " ${simdjson_VERSION})


# opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)


target_link_libraries(RoadLaneDetection ${OpenCV_LIBS} ${Boost_LIBRARIES})