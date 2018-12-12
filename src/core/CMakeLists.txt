#
#  KONA
#  CORE
#

add_library(core-${VERSION} STATIC
            "statis.c" "utili.c" 
            "para.c" "ops.c" 
	    "matvec.c" "orth.c")

install(TARGETS core-${VERSION} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY})
# try to let .h be here and install to include
#install(FILES "assemble.fd" "fem2Dtri.fd" DESTINATION ${INCLUDE_OUTPUT_DIRECTORY})
