interface TreeNode {
  feature?: number;
  threshold?: number;
  left?: TreeNode;
  right?: TreeNode;
  value?: number;
  isLeaf: boolean;
}

interface DecisionTreeParams {
  maxDepth?: number;
  minSamplesSplit?: number;
  minSamplesLeaf?: number;
}

export async function trainTree(data: { features: number[][]; targets: number[] }, params: DecisionTreeParams = {}): Promise<TreeNode> {
  const { features, targets } = data;
  const maxDepth = params.maxDepth ?? 10;
  const minSamplesSplit = params.minSamplesSplit ?? 2;
  const minSamplesLeaf = params.minSamplesLeaf ?? 1;
  return buildTree(features, targets, 0, maxDepth, minSamplesSplit, minSamplesLeaf);
}

function buildTree(X: number[][], y: number[], depth: number, maxDepth: number, minSplit: int, minLeaf: int): TreeNode {
  if (depth >= maxDepth || y.length < minSplit || new Set(y).size === 1) {
    return { isLeaf: true, value: y.reduce((a, b) => a + b, 0) / y.length };
  }
  let bestFeature = 0, bestThreshold = 0, bestGain = -Infinity;
  const totalVariance = variance(y);
  for (let f = 0; f < X[0].length; f++) {
    const values = [...new Set(X.map(row => row[f]))].sort((a, b) => a - b);
    for (const t of values) {
      const leftIdx: number[] = [], rightIdx: number[] = [];
      for (let i = 0; i < X.length; i++) {
        (X[i][f] <= t ? leftIdx : rightIdx).push(i);
      }
      if (leftIdx.length < minLeaf || rightIdx.length < minLeaf) continue;
      const leftY = leftIdx.map(i => y[i]);
      const rightY = rightIdx.map(i => y[i]);
      const gain = totalVariance - (leftY.length / y.length) * variance(leftY) - (rightY.length / y.length) * variance(rightY);
      if (gain > bestGain) { bestGain = gain; bestFeature = f; bestThreshold = t; }
    }
  }
  if (bestGain <= 0) return { isLeaf: true, value: y.reduce((a, b) => a + b, 0) / y.length };
  const leftIdx: number[] = [], rightIdx: number[] = [];
  for (let i = 0; i < X.length; i++) {
    (X[i][bestFeature] <= bestThreshold ? leftIdx : rightIdx).push(i);
  }
  return {
    isLeaf: false,
    feature: bestFeature,
    threshold: bestThreshold,
    left: buildTree(leftIdx.map(i => X[i]), leftIdx.map(i => y[i]), depth + 1, maxDepth, minSplit, minLeaf),
    right: buildTree(rightIdx.map(i => X[i]), rightIdx.map(i => y[i]), depth + 1, maxDepth, minSplit, minLeaf)
  };
}

function variance(arr: number[]): number {
  const mean = arr.reduce((a, b) => a + b, 0) / arr.length;
  return arr.reduce((s, v) => s + (v - mean) ** 2, 0) / arr.length;
}

export async function predict(tree: TreeNode, features: number[][]): Promise<number[]> {
  return features.map(row => predictOne(tree, row));
}

function predictOne(node: TreeNode, row: number[]): number {
  if (node.isLeaf) return node.value!;
  return row[node.feature!] <= node.threshold!
    ? predictOne(node.left!, row)
    : predictOne(node.right!, row);
}

export async function processItem(input: string, params: any = {}): Promise<any> {
  const parsed = JSON.parse(input);
  const { features, targets, predict_features } = parsed;
  if (!features || !targets) throw new Error("features and targets required");
  const tree = await trainTree({ features, targets }, params);
  const predictions = predict_features ? await predict(tree, predict_features) : [];
  return {
    tree_depth: getDepth(tree),
    leaf_count: getLeafCount(tree),
    predictions,
    training_samples: targets.length,
    timestamp: new Date().toISOString()
  };
}

function getDepth(node: TreeNode): number {
  if (node.isLeaf) return 0;
  return 1 + Math.max(getDepth(node.left!), getDepth(node.right!));
}

function getLeafCount(node: TreeNode): number {
  if (node.isLeaf) return 1;
  return getLeafCount(node.left!) + getLeafCount(node.right!);
}
