#include "DomTree.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"

static llvm::cl::opt<std::string>
    start("start", llvm::cl::desc("The file your want to bisect"),
          llvm::cl::init(""));

static llvm::cl::opt<bool> shouldDumpIDom("dump-idom",
                                          llvm::cl::desc("Dump the idom"),
                                          llvm::cl::Hidden);

mlir::ModuleOp loadMLIR(mlir::MLIRContext &context) {
  llvm::StringRef filename = ".bisect.mlir";
  if (start != "")
    filename = start;

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
  return mod.release();
}

int execute() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::BuiltinDialect>();
  context.allowUnregisteredDialects();

  mlir::ModuleOp mod = loadMLIR(context);
  if (!mod)
    return -1;

  IDomMapT idomMap = calcIDom(*mod.getOps<mlir::func::FuncOp>().begin());
  if (shouldDumpIDom) {
    dumpIDom(idomMap);
    return 0;
  }

  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  return execute();
}
