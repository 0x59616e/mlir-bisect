add_executable(mlir-bisect
    main.cpp
    DomTree.cpp)

target_link_libraries(mlir-bisect
    MLIRIR
    LLVMSupport
    MLIRParser
    MLIRFuncDialect
    MLIRArithDialect)

