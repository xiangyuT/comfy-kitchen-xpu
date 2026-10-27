cmake_minimum_required(VERSION 3.26)
project(comfy_kitchen_xpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED Development.SABIModule)
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
find_package(nanobind CONFIG REQUIRED)

set(SOURCES dlpack_bindings.cpp ops/int8_convrot.cpp)
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
    nanobind_add_module(_C NB_STATIC STABLE_ABI ${SOURCES})
    if(WIN32)
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.pyd")
    else()
        set_target_properties(_C PROPERTIES SUFFIX ".abi3.so")
    endif()
    target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x030C0000)
else()
    nanobind_add_module(_C NB_STATIC ${SOURCES})
endif()

target_compile_options(_C PRIVATE -fsycl -O3 -ffp-contract=off)
target_link_options(_C PRIVATE -fsycl)
set_target_properties(_C PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
install(TARGETS _C LIBRARY DESTINATION . RUNTIME DESTINATION .)
