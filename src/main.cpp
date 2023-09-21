#include "Bisect.h"
#include "DomTree.h"
#include "OpMarker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/raw_ostream.h"
#include <system_error>

static llvm::cl::opt<bool> start(
    "start",
    llvm::cl::desc(
        "The flag you should specify when doing bisect at the first time"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> good("good", llvm::cl::init(false));
static llvm::cl::opt<bool> bad("bad", llvm::cl::init(false));

static llvm::cl::opt<std::string> inputFile(llvm::cl::Positional,
                                            llvm::cl::desc("<input file>"),
                                            llvm::cl::init("-"));

static llvm::cl::opt<bool> shouldDumpIDom("dump-idom",
                                          llvm::cl::desc("Dump the idom"),
                                          llvm::cl::Hidden);

static llvm::cl::opt<std::string>
    outputFile("o", llvm::cl::desc("<output file>"), llvm::cl::init("-"));

mlir::ModuleOp loadMLIR(mlir::MLIRContext &context) {
  std::string filename;
  if (start || shouldDumpIDom)
    filename = inputFile;
  else
    filename = inputFile + ".bisect";

  mlir::OwningOpRef<mlir::ModuleOp> mod =
      mlir::parseSourceFile<mlir::ModuleOp>(filename, &context);
  return mod.release();
}

static void markReturnValueAsChecking(mlir::ModuleOp mod) {
  auto func = *mod.getOps<mlir::func::FuncOp>().begin();
  mlir::Block &blk = *func.getBlocks().begin();
  auto retOp = blk.getTerminator();
  markAsChecking(retOp->getOperand(0));
}

static void saveToFile(mlir::ModuleOp mod, llvm::StringRef filename) {
  std::error_code ec;
  llvm::raw_fd_ostream os(filename, ec);
  if (ec) {
    llvm::errs() << ec.message() << '\n';
    return;
  }

  mod.print(os);
}

static void saveBisectTmpFile(mlir::ModuleOp mod) {
  std::string outputFile = inputFile + ".bisect";
  saveToFile(mod, outputFile);
}

static void rewriteReturnOp(mlir::ModuleOp mod, mlir::Value value) {
  auto func = *mod.getOps<mlir::func::FuncOp>().begin();
  mlir::Block &blk = func.front();
  auto retOp = blk.getTerminator();

  retOp->setOperand(0, value);
  llvm::SmallVector<mlir::Type> resultTypes{value.getType()};
  auto funcType = mlir::FunctionType::get(
      mod.getContext(), func.getFunctionType().getInputs(), resultTypes);
  func.setFunctionType(funcType);
}

static void eliminateDeadOp(mlir::ModuleOp mod) {
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;

    mod.walk([&](mlir::Operation *op) {
      if (!op->getNumResults())
        return;
      bool isDead = llvm::all_of(op->getResults(), [&](mlir::OpResult result) {
        return result.getUses().empty();
      });

      if (isDead) {
        op->erase();
        hasChanged = true;
      }
    });
  }
}

static void saveMLIR(mlir::ModuleOp mod, SearchStatus status) {
  saveBisectTmpFile(mod);

  mlir::Value value;
  if (status == SearchStatus::FoundSuccessfully)
    value = getCulpritValue(mod);
  else if (status == SearchStatus::YetToBeFound)
    value = getCheckingValue(mod);

  rewriteReturnOp(mod, value);
  eliminateDeadOp(mod);
  clearBisectStatus(mod);
  saveToFile(mod, outputFile);
}

static int execute() {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect>();
  context.loadDialect<mlir::BuiltinDialect>();
  context.loadDialect<mlir::arith::ArithDialect>();
  context.allowUnregisteredDialects();

  mlir::ModuleOp mod = loadMLIR(context);
  if (!mod) {
    return -1;
  }

  IDomMapT idomMap = calcIDom(*mod.getOps<mlir::func::FuncOp>().begin());
  if (shouldDumpIDom) {
    dumpIDom(idomMap, mod);
    return 0;
  }

  if (start)
    markReturnValueAsChecking(mod);
  else if (!good ^ bad) {
    llvm::errs() << "You should assign `good` or `bad`";
    return -1;
  }

  bool isSuccess = !start && good;

  mlir::Value checkedValue = getCheckingValue(mod);
  if (!checkedValue) {
    llvm::errs() << "You should specify --start first\n";
    return -1;
  }

  if (isSuccess)
    markAsSuccess(checkedValue);
  else
    markAsFailed(checkedValue);

  auto status = searchCulprit(mod, idomMap);
  if (status == SearchStatus::FoundSuccessfully)
    llvm::errs() << "Culprit found successfully. Bisect over\n";

  saveMLIR(mod, status);

  return 0;
}

int main(int argc, char **argv) {
  llvm::InitLLVM X(argc, argv);

  llvm::cl::ParseCommandLineOptions(argc, argv);

  return execute();
}
