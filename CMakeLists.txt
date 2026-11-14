cmake_minimum_required(VERSION 3.16)
project(treeclust CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra)

# Core library: index structures and clustering (no dependencies).
add_library(treeclust STATIC
  src/dataset.cpp
  src/index.cpp
  src/cluster.cpp)
target_include_directories(treeclust PUBLIC src)

# JSON for the HTTP API: system package (nlohmann-json3-dev) or download.
find_package(nlohmann_json 3.2 QUIET)
if(NOT nlohmann_json_FOUND)
  include(FetchContent)
  FetchContent_Declare(json
    URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
  FetchContent_MakeAvailable(json)
endif()
find_package(Threads REQUIRED)

add_executable(treeclust-server src/http.cpp src/server.cpp)
target_link_libraries(treeclust-server PRIVATE treeclust nlohmann_json::nlohmann_json Threads::Threads)
target_compile_definitions(treeclust-server PRIVATE
  TREECLUST_WEB_DIR="${CMAKE_CURRENT_SOURCE_DIR}/web")

add_executable(treeclust-bench bench/bench.cpp)
target_link_libraries(treeclust-bench PRIVATE treeclust)

add_executable(treeclust-test tests/test_main.cpp)
target_link_libraries(treeclust-test PRIVATE treeclust)
enable_testing()
add_test(NAME treeclust-test COMMAND treeclust-test)
