#
#  KONA
#  CORE
#

add_library(app_user-${VERSION} STATIC
            "app_user.c")

install(TARGETS app_user-${VERSION} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY})
# try to let .h be here and install to include
#install(FILES "assemble.fd" "fem2Dtri.fd" DESTINATION ${INCLUDE_OUTPUT_DIRECTORY})
