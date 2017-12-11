IF (APPLE)
    cmake_minimum_required(VERSION 3.4)
ELSE()
    cmake_minimum_required(VERSION 2.8)
ENDIF()

project(ctc_release)

IF (NOT APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")
ENDIF()

IF (APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    add_definitions(-DAPPLE)
ENDIF()

include_directories(include)

IF (APPLE)
    EXEC_PROGRAM(uname ARGS -v  OUTPUT_VARIABLE DARWIN_VERSION)
    STRING(REGEX MATCH "[0-9]+" DARWIN_VERSION ${DARWIN_VERSION})
    MESSAGE(STATUS "DARWIN_VERSION=${DARWIN_VERSION}")

    #for el capitain have to use rpath

    IF (DARWIN_VERSION LESS 15)
        set(CMAKE_SKIP_RPATH TRUE)
    ENDIF ()

ELSE()
    #always skip for linux
    set(CMAKE_SKIP_RPATH TRUE)
ENDIF()


    MESSAGE(STATUS "Building shared library with no GPU support")

    if (NOT APPLE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")
    ENDIF()

    ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "bin"
            LIBRARY DESTINATION "lib"
            ARCHIVE DESTINATION "lib")

    INSTALL(FILES include/ctc.h DESTINATION "include")
