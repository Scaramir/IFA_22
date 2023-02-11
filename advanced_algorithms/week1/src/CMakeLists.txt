cmake_minimum_required (VERSION 3.8)

# A interface to reuse common properties.
# You can add more external include paths of other projects that are needed for your project.
add_library ("${PROJECT_NAME}_interface" INTERFACE)
target_link_libraries ("${PROJECT_NAME}_interface" INTERFACE seqan3::seqan3)
target_include_directories ("${PROJECT_NAME}_interface" INTERFACE ../include)
target_compile_options ("${PROJECT_NAME}_interface" INTERFACE "-pedantic" "-Wall" "-Wextra")

add_executable (naive_search naive_search.cpp)
target_link_libraries (naive_search PRIVATE "${PROJECT_NAME}_interface")

add_executable (fmindex_construct fmindex_construct.cpp)
target_link_libraries (fmindex_construct PRIVATE "${PROJECT_NAME}_interface")

add_executable (fmindex_search fmindex_search.cpp)
target_link_libraries (fmindex_search PRIVATE "${PROJECT_NAME}_interface")
target_link_libraries(fmindex_search PRIVATE "${PROJECT_NAME}_interface" benchmark::benchmark)

add_executable (fmindex_pigeon_search fmindex_pigeon_search.cpp)
target_link_libraries (fmindex_pigeon_search PRIVATE "${PROJECT_NAME}_interface")

add_executable (suffixarray_search suffixarray_search.cpp)
target_include_directories(suffixarray_search PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/../lib/libdivsufsort/include")
target_link_libraries (suffixarray_search PRIVATE "${PROJECT_NAME}_interface" divsufsort)
