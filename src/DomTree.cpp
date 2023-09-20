#include "DomTree.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "domtree"

using DominatorSetT = llvm::DenseSet<mlir::Value>;
using ValueToDominatorSetMapT = llvm::DenseMap<mlir::Value, DominatorSetT>;

static void dumpDominators(ValueToDominatorSetMapT &dominatorMap) {
  llvm::outs() << "********** Dominators **********\n";

  for (auto &[value, dominators] : dominatorMap) {
    value.dump();
    for (mlir::Value dominator : dominators) {
      llvm::outs() << "    ";
      dominator.dump();
    }
  }
}

void dumpIDom(const IDomMapT &idomMap) {
  for (auto [value, _] : idomMap) {
    value.print(llvm::outs());
    llvm::outs() << "\n";
  }
  llvm::outs() << "---\n";
  for (auto [value, dominator] : idomMap) {
    dominator.print(llvm::outs());
    llvm::outs() << " => ";
    value.print(llvm::outs());
    llvm::outs() << "\n";
  }
}

static std::vector<mlir::Value> getAllValues(mlir::func::FuncOp func) {
  auto arguments = func.getArguments();
  assert(arguments.size() >= 1);
  assert(arguments.size() == 1 &&
         "Function with more one argument is unsupported");

  std::vector<mlir::Value> allValues{arguments[0]};

  func.walk([&](mlir::Operation *op) {
    if (llvm::isa<mlir::arith::ConstantOp, mlir::func::FuncOp>(op))
      return;

    for (auto result : op->getResults())
      allValues.push_back(result);
  });

  return allValues;
}

static void calcUnion(DominatorSetT &set, DominatorSetT &other) {
  llvm::SmallVector<mlir::Value> valueToErase;
  for (auto value : set) {
    if (!other.contains(value))
      valueToErase.push_back(value);
  }
  for (mlir::Value value : valueToErase)
    set.erase(value);
}

static ValueToDominatorSetMapT
initializeValueToDominatorSetMap(llvm::ArrayRef<mlir::Value> allValues) {
  ValueToDominatorSetMapT dominatorMap;

  for (mlir::Value value : allValues) {
    if (llvm::isa<mlir::BlockArgument>(value)) {
      dominatorMap[value].insert(value);
      continue;
    }

    llvm::DenseSet<mlir::Value> &dominator = dominatorMap[value];
    for (mlir::Value other : allValues)
      dominator.insert(other);
  }

  return dominatorMap;
}

static bool isRoot(mlir::Value value) {
  return llvm::isa<mlir::BlockArgument>(value);
}

static bool calcUnionOfPreds(mlir::Value value,
                             ValueToDominatorSetMapT &dominatorMap) {
  mlir::Operation *op = value.getDefiningOp();
  assert(op->getNumOperands());
  llvm::DenseSet<mlir::Value> &dominator = dominatorMap[value];
  unsigned originalSize = dominator.size();

  for (mlir::Value pred : op->getOperands())
    calcUnion(dominator, dominatorMap[pred]);
  dominator.insert(value);

  return originalSize != dominator.size();
}

static void calcDominators(ValueToDominatorSetMapT &dominatorMap,
                           llvm::ArrayRef<mlir::Value> allValues) {
  bool hasChanged = true;
  while (hasChanged) {
    hasChanged = false;

    for (mlir::Value value : allValues) {
      if (isRoot(value))
        continue;

      hasChanged |= calcUnionOfPreds(value, dominatorMap);
    }
  }

  LLVM_DEBUG(dumpDominators(dominatorMap));
}

static bool isAStrictlyDominateB(mlir::Value A, mlir::Value B,
                                 ValueToDominatorSetMapT const &dominatorMap) {
  bool result = dominatorMap.find(B)->second.contains(A);
  return result;
}

static IDomMapT calcIDom(ValueToDominatorSetMapT &dominatorMap) {
  IDomMapT idomMap;

  for (auto &[value, dominators] : dominatorMap) {
    mlir::Value idom;
    auto it = llvm::find_if(
        dominators, [&](mlir::Value dominator) { return dominator != value; });

    if (it != dominators.end()) {
      idom = *it;
      for (auto dominator : dominators) {
        if (dominator != value &&
            !isAStrictlyDominateB(dominator, idom, dominatorMap)) {
          idom = dominator;
        }
      }
    }

    idomMap.insert({value, idom});
  }

  return idomMap;
}

IDomMapT calcIDom(mlir::func::FuncOp func) {
  std::vector<mlir::Value> allValues = getAllValues(func);
  auto dominatorMap = initializeValueToDominatorSetMap(allValues);
  calcDominators(dominatorMap, allValues);

  return calcIDom(dominatorMap);
}
