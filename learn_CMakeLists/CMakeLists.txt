cmake_minimum_required(VERSION 3.10.2)
project(CPP_Learning)

option(BUILD_TEST2 "Turn on build for tests of CPP_Learning" OFF)
# set variable BUILD_TEST2 as OFF as default

file(GLOB_RECURSE SRC_FILES
	"src/*.cpp"
)
# GLOB_RECURSE: glob .cpp recursively
# SRC_FILES = "src/main.cpp src/test1/test1.cpp"

set(INC_FILES_DIR
	include/
)
# INC_FILES_DIR = "include/"

if (BUILD_TESTS)
	add_subdirectory(src/test1)
endif()
# also build the src/test1/ if we set it as on

include_directories(
	${INC_FILES_DIR}
)

add_executable(CPP_Learning
	${SRC_FILES}
)

#target_link_libraries(CPP_Learning test1)

