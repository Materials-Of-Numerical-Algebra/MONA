#
#  KONA
#  CORE
#

add_library(linsol-${VERSION} STATIC
            "cg.c")

install(TARGETS linsol-${VERSION} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY})
# try to let .h be here and install to include
#install(FILES "assemble.fd" "fem2Dtri.fd" DESTINATION ${INCLUDE_OUTPUT_DIRECTORY})
