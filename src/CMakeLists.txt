add_executable(mlir-bisect
    main.cpp
    OpMarker.cpp
    Bisect.cpp)

target_link_libraries(mlir-bisect
    MLIRIR
    LLVMSupport
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect
    MLIRTransforms)

