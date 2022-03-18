
#[[

Exports target `GINKGO`

Users may set the following variables:

- HIOP_GINKGO_DIR

]]

find_package(Ginkgo CONFIG
    PATHS ${GINKGO_DIR} ${HIOP_GINKGO_DIR}
    REQUIRED)
    
