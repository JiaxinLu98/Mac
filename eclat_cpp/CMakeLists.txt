cmake_minimum_required(VERSION 3.0...3.25)
project(EclatGPU LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)
FetchContent_Declare(
    webgpu
    GIT_REPOSITORY https://github.com/eliemichel/WebGPU-distribution
    GIT_TAG        wgpu-v24.0.0.2
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(webgpu)

# main.cpp (the unoptimized harness) is not part of the M5 Pro package, only AppOpt is run there.
set(ECLAT_TARGETS AppOpt)
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
    add_executable(App main.cpp)
    target_link_libraries(App PRIVATE webgpu)
    target_copy_webgpu_binaries(App)
    list(PREPEND ECLAT_TARGETS App)
endif()

add_executable(AppOpt main_opt.cpp)
target_link_libraries(AppOpt PRIVATE webgpu)
target_copy_webgpu_binaries(AppOpt)

set(SHADER_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/balanced_path_batched.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/set_availability_decoupled_lookback_batched_fulloutput.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/balanced_path_batched_pairinfo.wgsl
    ${CMAKE_CURRENT_SOURCE_DIR}/shaders/set_availability_decoupled_lookback_batched_fulloutput_pairinfo.wgsl
)

foreach(SHADER ${SHADER_FILES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    configure_file(${SHADER} ${CMAKE_CURRENT_BINARY_DIR}/shaders/${SHADER_NAME} COPYONLY)
endforeach()

foreach(TARGET_NAME ${ECLAT_TARGETS})
    add_custom_command(TARGET ${TARGET_NAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:${TARGET_NAME}>/shaders"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${SHADER_FILES}
            "$<TARGET_FILE_DIR:${TARGET_NAME}>/shaders/"
        COMMENT "Copying shader files to output directory"
    )
endforeach()
