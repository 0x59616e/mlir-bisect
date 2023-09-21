#include "Bisect.h"
#include "DomTree.h"
#include "OpMarker.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/SmallVector.h"

#include <queue>

static mlir::Value getReturnValue(mlir::ModuleOp mod) {
  auto func = *mod.getOps<mlir::func::FuncOp>().begin();
  mlir::Block &blk = func.front();
  auto retOp = blk.getTerminator();
  return retOp->getOperand(0);
}

llvm::SmallVector<mlir::Value> getDominators(mlir::Value value,
                                             const IDomMapT &idomMap) {
  llvm::SmallVector<mlir::Value> dominators{value};

  while (true) {
    auto it = idomMap.find(value);
    if (it == idomMap.end())
      break;

    value = it->second;
    dominators.push_back(value);
  }

  return dominators;
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
  if (llvm::isa<mlir::BlockArgument>(value))
    return false;

  mlir::Operation *op = value.getDefiningOp();

  for (auto operand : op->getOperands()) {
    if (isFailedOrUnknown(operand))
      return false;
  }

  return true;
}

SearchStatus searchCulprit(mlir::ModuleOp mod, const IDomMapT &idomMap) {
  std::queue<mlir::Value> searchQueue;
  searchQueue.push(getReturnValue(mod));

  while (!searchQueue.empty()) {
    mlir::Value value = searchQueue.front();
    searchQueue.pop();
    if (isUnknown(value)) {
      markAsChecking(value);
      return SearchStatus::YetToBeFound;
    }

    auto dominators = getDominators(value, idomMap);
    mlir::Value possibleCulprit = doBinarySearch(dominators);
    if (isUnknown(possibleCulprit)) {
      markAsChecking(possibleCulprit);
      return SearchStatus::YetToBeFound;
    }

    if (isThisValueCulprit(possibleCulprit)) {
      markAsCulprit(possibleCulprit);
      return SearchStatus::FoundSuccessfully;
    }

    for (auto pred : value.getDefiningOp()->getOperands())
      if (isFailedOrUnknown(pred))
        searchQueue.push(pred);
  }

  llvm_unreachable("unimplemented");
}