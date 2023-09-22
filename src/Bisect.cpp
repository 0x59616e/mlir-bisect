#include "Bisect.h"
#include "DomTree.h"
#include "OpMarker.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"

static mlir::Value getReturnValue(mlir::ModuleOp mod) {
  auto func = *mod.getOps<mlir::func::FuncOp>().begin();
  mlir::Block &blk = func.front();
  auto retOp = blk.getTerminator();
  return retOp->getOperand(0);
}

static bool getValuesToSearch(mlir::Value value,
                              std::vector<mlir::Value> &valuesToSearch) {
  assert(isFailedOrUnknown(value));

  mlir::Operation *op = value.getDefiningOp();
  if (llvm::isa<mlir::arith::ConstantOp>(op))
    return false;
  assert(op->getNumOperands());

  valuesToSearch.push_back(value);

  for (auto operand : op->getOperands()) {
    if (isSuccess(operand)) {
      valuesToSearch.push_back(operand);
      return true;
    }
    if (isFailedOrUnknown(operand) &&
        getValuesToSearch(operand, valuesToSearch))
      return true;
  }

  return false;
}

static std::vector<mlir::Value> getValuesToSearch(mlir::Value value) {
  std::vector<mlir::Value> valuesToSearch;
  assert(getValuesToSearch(value, valuesToSearch));
  return valuesToSearch;
}

static mlir::Value doBinarySearch(llvm::ArrayRef<mlir::Value> dominators) {
  unsigned exclusiveStart = 0;
  unsigned inclusiveEnd = dominators.size() - 1;

  while (inclusiveEnd - exclusiveStart > 1) {
    unsigned middle = (exclusiveStart + inclusiveEnd) / 2;
    if (isFailed(dominators[middle])) {
      exclusiveStart = middle;
    } else if (isSuccess(dominators[middle])) {
      inclusiveEnd = middle;
    } else {
      return dominators[middle];
      break;
    }
  }

  return dominators[exclusiveStart];
}

static bool isThisValueCulprit(mlir::Value value) {
  assert(isFailed(value));

  if (llvm::isa<mlir::BlockArgument>(value))
    return false;

  mlir::Operation *op = value.getDefiningOp();

  return llvm::all_of(op->getOperands(),
                      [](mlir::Value operand) { return isSuccess(operand); });
}

mlir::Value searchCulprit(mlir::Value value, const IDomMapT &idomMap) {
  if (isUnknown(value))
    return value;

  auto valuesToSearch = getValuesToSearch(value);
  mlir::Value possibleCulprit = doBinarySearch(valuesToSearch);

  if (isUnknown(possibleCulprit) || isThisValueCulprit(possibleCulprit))
    return possibleCulprit;

  for (auto operand : possibleCulprit.getDefiningOp()->getOperands())
    if (isFailed(operand))
      return searchCulprit(operand, idomMap);

  for (auto operand : possibleCulprit.getDefiningOp()->getOperands())
    if (isUnknown(operand))
      return searchCulprit(operand, idomMap);

  llvm_unreachable("should not be here");
}

SearchStatus searchCulprit(mlir::ModuleOp mod, const IDomMapT &idomMap) {
  mlir::Value value = searchCulprit(getReturnValue(mod), idomMap);
  if (isUnknown(value)) {
    markAsChecking(value);
    return SearchStatus::YetToBeFound;
  }

  if (isFailed(value)) {
    markAsCulprit(value);
    return SearchStatus::FoundSuccessfully;
  }

  llvm_unreachable("should not be here");
}