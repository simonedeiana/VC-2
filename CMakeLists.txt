cmake_minimum_required(VERSION 3.21)

project(vc2_windows LANGUAGES CXX)

if(NOT MSVC)
  message(FATAL_ERROR "This build is intentionally for the Microsoft Visual C++ toolchain.")
endif()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
foreach(config Debug Release RelWithDebInfo MinSizeRel)
  string(TOUPPER "${config}" config_upper)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/bin")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib")
endforeach()

add_compile_options(/W3 /EHsc /permissive- /Zc:__cplusplus)
add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/msvc-compat")

find_package(Boost REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter)
include(CTest)

# VC-2 reference encoder/decoder and reference library.
set(reference_root "${CMAKE_CURRENT_SOURCE_DIR}/vc2-reference")
set(reference_sources
  ${reference_root}/src/Library/src/Arrays.cpp
  ${reference_root}/src/Library/src/DataUnit.cpp
  ${reference_root}/src/Library/src/Frame.cpp
  ${reference_root}/src/Library/src/Picture.cpp
  ${reference_root}/src/Library/src/Quantisation.cpp
  ${reference_root}/src/Library/src/Slices.cpp
  ${reference_root}/src/Library/src/Utils.cpp
  ${reference_root}/src/Library/src/VLC.cpp
  ${reference_root}/src/Library/src/WaveletTransform.cpp)
add_library(vc2reference STATIC ${reference_sources})
target_include_directories(vc2reference PUBLIC
  ${reference_root}/src
  ${reference_root}/src/Library
  ${Boost_INCLUDE_DIRS})

add_executable(EncodeStream
  ${reference_root}/src/EncodeStream/EncodeStream.cpp
  ${reference_root}/src/EncodeStream/EncodeParams.cpp)
target_include_directories(EncodeStream PRIVATE ${reference_root}/src/tclap)
target_link_libraries(EncodeStream PRIVATE vc2reference)

add_executable(DecodeStream
  ${reference_root}/src/DecodeStream/DecodeStream.cpp
  ${reference_root}/src/DecodeStream/DecodeParams.cpp)
target_include_directories(DecodeStream PRIVATE ${reference_root}/src/tclap)
target_link_libraries(DecodeStream PRIVATE vc2reference)

# High-performance HQ encoder.
set(encode_root "${CMAKE_CURRENT_SOURCE_DIR}/vc2hqencode")
set(encode_generated "${CMAKE_CURRENT_BINARY_DIR}/generated/vc2hqencode")
file(MAKE_DIRECTORY "${encode_generated}/avx" "${encode_generated}/avx2")
foreach(variant avx avx2)
  add_custom_command(
    OUTPUT
      "${encode_generated}/${variant}/haar_transform_sse4_2.hpp"
      "${encode_generated}/${variant}/legall_transform_sse4_2.hpp"
    COMMAND ${Python3_EXECUTABLE} ${encode_root}/duplicate-transform
      ${encode_root}/vc2transform_sse4_2/haar_transform.hpp
      "${encode_generated}/${variant}/haar_transform_sse4_2.hpp"
      sse4_2 sse4_2_${variant}
    COMMAND ${Python3_EXECUTABLE} ${encode_root}/duplicate-transform
      ${encode_root}/vc2transform_sse4_2/legall_transform.hpp
      "${encode_generated}/${variant}/legall_transform_sse4_2.hpp"
      sse4_2 sse4_2_${variant}
    DEPENDS
      ${encode_root}/duplicate-transform
      ${encode_root}/vc2transform_sse4_2/haar_transform.hpp
      ${encode_root}/vc2transform_sse4_2/legall_transform.hpp
    VERBATIM)
endforeach()

add_library(vc2transform_c STATIC ${encode_root}/vc2transform_c/transform_c.cpp)
add_library(vc2transform_sse STATIC ${encode_root}/vc2transform_sse4_2/transform_sse4_2.cpp)
add_library(vc2transform_avx STATIC
  ${encode_root}/vc2transform_avx/transform_avx.cpp
  "${encode_generated}/avx/haar_transform_sse4_2.hpp"
  "${encode_generated}/avx/legall_transform_sse4_2.hpp")
add_library(vc2transform_avx2 STATIC
  ${encode_root}/vc2transform_avx2/transform_avx2.cpp
  "${encode_generated}/avx2/haar_transform_sse4_2.hpp"
  "${encode_generated}/avx2/legall_transform_sse4_2.hpp")
foreach(target vc2transform_c vc2transform_sse vc2transform_avx vc2transform_avx2)
  target_include_directories(${target} PUBLIC ${encode_root} PRIVATE ${encode_root}/vc2hqencode)
endforeach()
target_include_directories(vc2transform_avx PRIVATE "${encode_generated}/avx")
target_include_directories(vc2transform_avx2 PRIVATE "${encode_generated}/avx2")
target_compile_options(vc2transform_avx PRIVATE /arch:AVX)
target_compile_options(vc2transform_avx2 PRIVATE /arch:AVX2)

add_library(vc2hqencode SHARED
  ${encode_root}/vc2hqencode/vc2hqencode.cpp
  ${encode_root}/vc2hqencode/VC2Encoder.cpp
  ${encode_root}/vc2hqencode/logger.cpp
  ${encode_root}/vc2hqencode/quantise.cpp
  ${encode_root}/vc2hqencode/encode.cpp
  ${encode_root}/vc2hqencode/stats.cpp
  ${encode_root}/vc2hqencode/stream.cpp)
set_property(TARGET vc2hqencode PROPERTY CXX_STANDARD 20)
target_compile_definitions(vc2hqencode PRIVATE
  VC2HQENCODE_DLL ASIO_STANDALONE _HAS_DEPRECATED_RESULT_OF=1)
target_include_directories(vc2hqencode PUBLIC ${encode_root} PRIVATE ${encode_root}/redist)
target_link_libraries(vc2hqencode PRIVATE
  vc2transform_c vc2transform_sse vc2transform_avx vc2transform_avx2)

add_executable(vc2encode
  ${encode_root}/testprogs/vc2encode.cpp
  ${encode_root}/testprogs/movingzoneplate.cpp)
target_include_directories(vc2encode PRIVATE ${encode_root} ${encode_root}/testprogs)
target_link_libraries(vc2encode PRIVATE vc2hqencode)

# High-performance HQ decoder.
set(decode_root "${CMAKE_CURRENT_SOURCE_DIR}/vc2hqdecode")
add_library(vc2invtransform_c STATIC
  ${decode_root}/vc2inversetransform_c/invtransform_c.cpp
  ${decode_root}/vc2inversetransform_c/dequantise_c.cpp
  ${decode_root}/vc2inversetransform_c/vlc_c.cpp)
add_library(vc2invtransform_sse STATIC
  ${decode_root}/vc2inversetransform_sse4_2/invtransform_sse4_2.cpp
  ${decode_root}/vc2inversetransform_sse4_2/dequantise_sse4_2.cpp
  ${decode_root}/vc2inversetransform_sse4_2/vlc_sse4_2.cpp)
foreach(target vc2invtransform_c vc2invtransform_sse)
  target_include_directories(${target} PUBLIC ${decode_root} PRIVATE ${decode_root}/vc2hqdecode)
endforeach()

add_library(vc2hqdecode SHARED
  ${decode_root}/vc2hqdecode/vc2hqdecode.cpp
  ${decode_root}/vc2hqdecode/logger.cpp
  ${decode_root}/vc2hqdecode/VC2Decoder.cpp
  ${decode_root}/vc2hqdecode/quantmatrix.cpp
  ${decode_root}/vc2hqdecode/stream.cpp)
target_compile_definitions(vc2hqdecode PRIVATE VC2HQDECODE_DLL ASIO_STANDALONE)
target_include_directories(vc2hqdecode PUBLIC ${decode_root} PRIVATE ${decode_root}/redist)
target_link_libraries(vc2hqdecode PRIVATE vc2invtransform_c vc2invtransform_sse)

add_executable(vc2decode ${decode_root}/testprogs/vc2decode.cpp)
target_include_directories(vc2decode PRIVATE ${decode_root} ${decode_root}/testprogs)
target_link_libraries(vc2decode PRIVATE vc2hqdecode)

if(BUILD_TESTING)
  add_library(vc2_gtest STATIC ${reference_root}/googletest/googletest/src/gtest-all.cc)
  target_include_directories(vc2_gtest PUBLIC
    ${reference_root}/googletest/googletest/include
    ${reference_root}/googletest/googletest)
  foreach(test_name Arrays Quantisation DataUnit Utils)
    string(TOLOWER "${test_name}" test_target)
    add_executable(reference_${test_target} ${reference_root}/tests/${test_name}.cpp)
    target_link_libraries(reference_${test_target} PRIVATE vc2reference vc2_gtest)
    add_test(NAME reference_${test_target} COMMAND reference_${test_target})
  endforeach()

  add_executable(vc2encodertest
    ${encode_root}/testsuite/tests.cpp
    ${encode_root}/testsuite/test_transforms.cpp
    ${encode_root}/vc2hqencode/logger.cpp)
  target_include_directories(vc2encodertest PRIVATE ${encode_root} ${encode_root}/vc2hqencode)
  target_link_libraries(vc2encodertest PRIVATE
    vc2transform_c vc2transform_sse vc2transform_avx vc2transform_avx2)
  add_test(NAME vc2encodertest COMMAND vc2encodertest)

  add_executable(vc2decodertest
    ${decode_root}/testsuite/tests.cpp
    ${decode_root}/testsuite/test_invtransform.cpp
    ${decode_root}/testsuite/test_dequantise.cpp
    ${decode_root}/testsuite/randomiser.cpp
    ${decode_root}/vc2hqdecode/quantmatrix.cpp
    ${decode_root}/vc2hqdecode/logger.cpp)
  target_compile_definitions(vc2decodertest PRIVATE VC2HQDECODE_DLL)
  target_include_directories(vc2decodertest PRIVATE
    ${decode_root} ${decode_root}/vc2hqdecode ${decode_root}/redist)
  target_link_libraries(vc2decodertest PRIVATE vc2invtransform_c vc2invtransform_sse)
  add_test(NAME vc2decodertest COMMAND vc2decodertest)
endif()

install(TARGETS vc2reference vc2hqencode vc2hqdecode EncodeStream DecodeStream vc2encode vc2decode
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION bin
  ARCHIVE DESTINATION lib)
install(FILES
  ${encode_root}/vc2hqencode/vc2hqencode.h
  DESTINATION include/vc2hqencode)
install(FILES
  ${decode_root}/vc2hqdecode/vc2hqdecode.h
  ${decode_root}/vc2hqdecode/vc2hqdecodestrings.h
  DESTINATION include/vc2hqdecode)
