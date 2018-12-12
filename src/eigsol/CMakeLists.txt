#
#  KONA
#  CORE
#

add_library(eigsol-${VERSION} STATIC
            "eigen_solver.c" "lobpcg.c" 
            "rayleighritz.c" "xpw.c" 
	    "workspace.c")

install(TARGETS eigsol-${VERSION} DESTINATION ${LIBRARY_OUTPUT_DIRECTORY})
# try to let .h be here and install to include
#install(FILES "assemble.fd" "fem2Dtri.fd" DESTINATION ${INCLUDE_OUTPUT_DIRECTORY})
