cmake_minimum_required(VERSION 2.6)
project(KONA)
enable_language(Fortran)


# default parameters
set(CMAKE_BUILD_TYPE Release)
set(VERSION 1.0)


set(LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../lib)
set(INCLUDE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../include)
set(RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/../bin)
set(PROJECT_DIR ${PROJECT_BINARY_DIR}/..)

set(ADDITIONAL_MAKE_CLEAN_FILES build)



# process input parameters
if (ENABLE_SYSTEM_LAPACK)
  set(LAPACK_LIBS "lapack;blas")
else ()
  set(LAPACK_LIBS "lapack-3.2;lapack_ext-3.2;blas-3.2")
endif()


if (FIX_CMAKE_BUGS)
  set(CMAKE_SHARED_LIBRARY_LINK_C_FLAGS "")
endif()


# detect Platform
if (${CMAKE_SYSTEM_NAME} MATCHES "Windows") 
  set(SUFFICS_EXE "")
else ()
  set(SUFFICS_EXE ".exe")
endif()

if (${CMAKE_SYSTEM_NAME} MATCHES "SunOS")
# set(LIBRARY_G2C "g2c") Solaris 11 and earlier
  set(LIBRARY_G2C "")
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Windows") 
  set(LIBRARY_G2C "gfortran")
else ()
  set(LIBRARY_G2C "")
endif()


# version of TPLs
#set(UMFPACK_VER "5.1")
set(LAPACK_VER "3.2")


###########################################################
# include directories
###########################################################
include_directories(${PROJECT_BINARY_DIR}/../include/core)
include_directories(${PROJECT_BINARY_DIR}/../include/linsol)
include_directories(${PROJECT_BINARY_DIR}/../include/eigsol)
include_directories(${PROJECT_BINARY_DIR}/../include/app/user)
include_directories(${PROJECT_BINARY_DIR}/../external/user/include)
#include_directories(${PROJECT_BINARY_DIR}/../include/app/umfpack)
#include_directories(${PROJECT_BINARY_DIR}/../include/app/hypre)


# list libraries
add_subdirectory(src/core)
add_subdirectory(src/linsol)
add_subdirectory(src/eigsol)
#add_subdirectory(src/svdsol)
#add_subdirectory(src/lsqsol)
add_subdirectory(src/blas)
add_subdirectory(src/lapack)
add_subdirectory(src/lapack/extension)

add_subdirectory(app/user)
#add_subdirectory(app/umfpack)
#add_subdirectory(app/hypre)



###########################################################
# test packages
###########################################################
enable_testing()

add_subdirectory(test/user)
#add_subdirectory(test/umfpack)
#add_subdirectory(test/hypre)

add_test(core_user_ops     ${RUNTIME_OUTPUT_DIRECTORY}/kona_user_ops.exe)
#add_test(CORE_vec         ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_vec.exe)
#add_test(CORE_multivec    ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_multivec.exe)
#add_test(CORE_matvec      ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_matvec.exe)
#add_test(CORE_matmultivec ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_matmultivec.exe)
#add_test(CORE_orth        ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_orth.exe)
#add_test(CORE_linsol      ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_linsol.exe)
#add_test(CORE_eigsol      ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_eigsol.exe)
#add_test(CORE_svdsol      ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_svdsol.exe)
#add_test(CORE_lsqsol      ${RUNTIME_OUTPUT_DIRECTORY}/konaCORE_lsqsol.exe)

include (CTest)
