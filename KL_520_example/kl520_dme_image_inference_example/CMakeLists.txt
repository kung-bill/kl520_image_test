# build with current *.c/*.cpp plus common source files in parent folder
# executable name is current folder name.
# This example needs cmake '-DBUILD_OPENCV_EX=on'

if(BUILD_OPENCV_EX)

get_filename_component(app_name ${CMAKE_CURRENT_SOURCE_DIR} NAME)
string(REPLACE " " "_" app_name ${app_name})

file(GLOB cur_folder_src
    "*.c"
    "*.cpp"
	)

include_directories(../)
set(extra_src
	../main.cpp
	../user_util.cpp
	../post_processing_ex.c
	)

add_executable(${app_name}
	${cur_folder_src}
	${extra_src}
	)

target_link_libraries(${app_name} ${HOST_LIB} ${USB_LIB} ${OpenCV_LIBS} pthread)

endif()
