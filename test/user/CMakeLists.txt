set(USER_LINK_DIR ${PROJECT_BINARY_DIR}/../external/user/lib)

# Make sure the compiler can find include files
include_directories(${INCLUDE_OUTPUT_DIRECTORY})

# Make sure the linker can find the libraries once it is built
link_directories(${LIBRARY_OUTPUT_DIRECTORY} ${USER_LINK_DIR})

# Add first test
set(TEST1 kona_user_ops${SUFFICS_EXE})
add_executable(${TEST1} "kona_user_ops.c")
target_link_libraries(${TEST1} core-${VERSION} app_user-${VERSION} 
			       libuser.a
                               ${LAPACK_LIBS})
install(TARGETS ${TEST1} DESTINATION ${RUNTIME_OUTPUT_DIRECTORY})
