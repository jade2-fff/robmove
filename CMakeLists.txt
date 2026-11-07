cmake_minimum_required(VERSION 3.20)
project(robmove VERSION 0.1.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
option(ROBMOVE_SANITIZERS "Enable address and undefined-behavior sanitizers" OFF)
option(ROBMOVE_WARNINGS_AS_ERRORS "Treat project compiler warnings as errors" ON)
option(ROBMOVE_BUILD_BENCHMARKS "Build offline microbenchmarks" ON)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(nlohmann_json 3.2 REQUIRED)
find_package(Threads REQUIRED)

function(robmove_target target)
    target_compile_features(${target} PUBLIC cxx_std_20)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wconversion -Wshadow)
        if(ROBMOVE_WARNINGS_AS_ERRORS)
            target_compile_options(${target} PRIVATE -Werror)
        endif()
        if(ROBMOVE_SANITIZERS)
            target_compile_options(${target} PUBLIC -fsanitize=address,undefined -fno-omit-frame-pointer)
            target_link_options(${target} PUBLIC -fsanitize=address,undefined)
        endif()
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    endif()
endfunction()

add_library(robmove_core src/geometry.cpp src/planning.cpp src/config.cpp src/ports.cpp src/task.cpp src/workflow.cpp)
add_library(robmove::core ALIAS robmove_core)
target_include_directories(robmove_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(robmove_core PUBLIC Eigen3::Eigen Threads::Threads PRIVATE nlohmann_json::nlohmann_json)
robmove_target(robmove_core)

add_library(robmove_adapters src/adapters/replay.cpp src/adapters/simulated_robot.cpp)
add_library(robmove::adapters ALIAS robmove_adapters)
target_link_libraries(robmove_adapters PUBLIC robmove::core PRIVATE nlohmann_json::nlohmann_json)
robmove_target(robmove_adapters)

add_executable(robmove app/main.cpp)
target_link_libraries(robmove PRIVATE robmove::adapters nlohmann_json::nlohmann_json)
robmove_target(robmove)

if(ROBMOVE_BUILD_BENCHMARKS)
    add_executable(robmove_benchmark benchmarks/planning_benchmark.cpp)
    target_link_libraries(robmove_benchmark PRIVATE robmove::core nlohmann_json::nlohmann_json)
    robmove_target(robmove_benchmark)
endif()

if(BUILD_TESTING)
    add_executable(robmove_tests tests/test_main.cpp tests/geometry_test.cpp tests/planning_test.cpp tests/config_test.cpp tests/workflow_test.cpp tests/adapters_test.cpp tests/golden_test.cpp)
    target_link_libraries(robmove_tests PRIVATE robmove::adapters nlohmann_json::nlohmann_json)
    target_compile_definitions(robmove_tests PRIVATE ROBMOVE_TEST_DATA="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    robmove_target(robmove_tests)
    add_test(NAME core COMMAND robmove_tests)
    add_test(NAME installed_consumer COMMAND ${CMAKE_COMMAND}
        -DROBMOVE_BUILD=${CMAKE_CURRENT_BINARY_DIR}
        -DROBMOVE_SOURCE=${CMAKE_CURRENT_SOURCE_DIR}
        -DROBMOVE_COMPILER=${CMAKE_CXX_COMPILER}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/tests/package_smoke.cmake)
    set_tests_properties(installed_consumer PROPERTIES TIMEOUT 60)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME cli COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_test.py $<TARGET_FILE:robmove> ${CMAKE_CURRENT_SOURCE_DIR})
        set_tests_properties(cli PROPERTIES TIMEOUT 30)
        if(Python3_VERSION VERSION_GREATER_EQUAL "3.11")
            add_test(NAME legacy_integrity COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/legacy_integrity.py ${CMAKE_CURRENT_SOURCE_DIR})
        endif()
    endif()
endif()

set_target_properties(robmove_core PROPERTIES EXPORT_NAME core)
set_target_properties(robmove_adapters PROPERTIES EXPORT_NAME adapters)
install(TARGETS robmove_core robmove_adapters robmove EXPORT robmoveTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT robmoveTargets NAMESPACE robmove:: DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robmove)
include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/robmoveConfig.cmake.in ${CMAKE_CURRENT_BINARY_DIR}/robmoveConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robmove)
write_basic_package_version_file(${CMAKE_CURRENT_BINARY_DIR}/robmoveConfigVersion.cmake
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMajorVersion)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/robmoveConfig.cmake ${CMAKE_CURRENT_BINARY_DIR}/robmoveConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/robmove)
