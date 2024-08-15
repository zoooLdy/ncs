# SPDX-License-Identifier: Apache-2.0

# This file provides Zephyr sdk config version package functionality.
#

# Those are Zephyr variables used.
get_filename_component(NCS_TOOLCHAIN_PATH ${CMAKE_CURRENT_LIST_DIR}/../ ABSOLUTE)
if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
  set(NCS_TOOLCHAIN_EXECUTABLE_SUFFIX ".exe")
  set(NCS_TOOLCHAIN_BIN_PATH   ${NCS_TOOLCHAIN_PATH}/opt/bin)

  # This will be deprecated, relying on it may fail for NCS 2.0+
  set(NCS_GNUARMEMB_TOOLCHAIN_PATH ${NCS_TOOLCHAIN_PATH}/opt)

  set(NCS_TOOLCHAIN_ENV_PATH   "${NCS_TOOLCHAIN_BIN_PATH};${NCS_TOOLCHAIN_BIN_PATH}/Scripts;${NCS_TOOLCHAIN_PATH}/mingw64/bin")
  set(NCS_TOOLCHAIN_PYTHONPATH "${NCS_TOOLCHAIN_BIN_PATH}/;${NCS_TOOLCHAIN_BIN_PATH}/Lib;${NCS_TOOLCHAIN_BIN_PATH}/Lib/site-packages")
  set(NCS_TOOLCHAIN_PYTHON     ${NCS_TOOLCHAIN_BIN_PATH}/python.exe)
  set(NCS_TOOLCHAIN_WEST       ${NCS_TOOLCHAIN_BIN_PATH}/Scripts/west.exe)
  set(NCS_TOOLCHAIN_GIT        ${NCS_TOOLCHAIN_PATH}/bin/git.exe)
else()
  set(NCS_TOOLCHAIN_EXECUTABLE_SUFFIX "")
  if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Linux)
    set(NCS_TOOLCHAIN_BIN_PATH      ${NCS_TOOLCHAIN_PATH}/usr/local/bin)
    set(NCS_TOOLCHAIN_ENV_PATH      "${NCS_TOOLCHAIN_BIN_PATH};${NCS_TOOLCHAIN_PATH}/bin")
  else()
    set(NCS_TOOLCHAIN_BIN_PATH      ${NCS_TOOLCHAIN_PATH}/bin)
    set(NCS_TOOLCHAIN_ENV_PATH      ${NCS_TOOLCHAIN_BIN_PATH})
  endif()
  set(NCS_TOOLCHAIN_WEST            ${NCS_TOOLCHAIN_BIN_PATH}/west)
  set(NCS_TOOLCHAIN_PYTHON          ${NCS_TOOLCHAIN_BIN_PATH}/python3)

  # This will be deprecated, relying on it may fail for NCS 2.0+
  set(NCS_GNUARMEMB_TOOLCHAIN_PATH  ${NCS_TOOLCHAIN_PATH})

  set(NCS_TOOLCHAIN_GIT             ${NCS_TOOLCHAIN_BIN_PATH}/git)
endif()

set(NCS_TOOLCHAIN_CMAKE      ${NCS_TOOLCHAIN_BIN_PATH}/cmake${NCS_TOOLCHAIN_EXECUTABLE_SUFFIX})
set(NCS_TOOLCHAIN_NINJA      ${NCS_TOOLCHAIN_BIN_PATH}/ninja${NCS_TOOLCHAIN_EXECUTABLE_SUFFIX})
set(NCS_TOOLCHAIN_DTC        ${NCS_TOOLCHAIN_BIN_PATH}/dtc${NCS_TOOLCHAIN_EXECUTABLE_SUFFIX})
set(NCS_TOOLCHAIN_GPERF      ${NCS_TOOLCHAIN_BIN_PATH}/gperf${NCS_TOOLCHAIN_EXECUTABLE_SUFFIX})
set(NCS_TOOLCHAIN_PROTOC     ${NCS_TOOLCHAIN_PATH}/opt/nanopb/generator-bin/protoc${NCS_TOOLCHAIN_EXECUTABLE_SUFFIX})

set(NCS_TOOLCHAIN_VARIANT zephyr)
set(NCS_ZEPHYR_SDK_INSTALL_DIR ${NCS_TOOLCHAIN_PATH}/opt/zephyr-sdk)

# Those are CMake package parameters.
set(NcsToolchain_FOUND True)

if("${CMAKE_SCRIPT_MODE_FILE}" STREQUAL "${CMAKE_CURRENT_LIST_FILE}")
  if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
    set(TMP_TOOLCHAIN_PATH_FILE "${CMAKE_CURRENT_LIST_DIR}/NcsToolchainPath.txt")
  else()
    set(TMP_TOOLCHAIN_PATH_FILE "/tmp/NcsToolchainPath.txt")
  endif()
  file(WRITE ${TMP_TOOLCHAIN_PATH_FILE} ${CMAKE_CURRENT_LIST_DIR})
  execute_process(COMMAND ${CMAKE_COMMAND} -E md5sum ${TMP_TOOLCHAIN_PATH_FILE}
                  OUTPUT_VARIABLE MD5_SUM
  )
  string(SUBSTRING ${MD5_SUM} 0 32 MD5_SUM)
  if(${CMAKE_HOST_SYSTEM_NAME} STREQUAL Windows)
    execute_process(COMMAND ${CMAKE_COMMAND} -E write_regv
      "HKEY_CURRENT_USER\\Software\\Kitware\\CMake\\Packages\\NcsToolchain\\;${MD5_SUM}" "${CMAKE_CURRENT_LIST_DIR}"
    )
  else()
    file(WRITE $ENV{HOME}/.cmake/packages/NcsToolchain/${MD5_SUM} ${CMAKE_CURRENT_LIST_DIR})
  endif()
  file(REMOVE ${TMP_TOOLCHAIN_PATH_FILE})
endif()
