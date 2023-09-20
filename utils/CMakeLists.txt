configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/mlir-bisect-lit.in
    ${CMAKE_CURRENT_BINARY_DIR}/mlir-bisect-lit
    FILE_PERMISSIONS OWNER_EXECUTE OWNER_WRITE OWNER_READ
    @ONLY
)

