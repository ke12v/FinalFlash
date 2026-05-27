import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, ScatterChart, Scatter } from 'recharts';
import { TreeDeciduous, Users, Brain, Target, Zap, Info, Download, FileText } from 'lucide-react';
import { toast } from 'sonner';

function DecisionTreeVisualization({ treeStructure, featureImportance }) {
  const [selectedNode, setSelectedNode] = useState(null);

  const renderTreeNode = (node, depth = 0) => {
    if (!node) return null;

    if (node.is_leaf) {
      return (
        <div
          key={node.node_id}
          className={`ml-4 p-2 rounded border ${
            selectedNode === node.node_id ? 'bg-blue-100 border-blue-500' : 'bg-green-50 border-green-300'
          } cursor-pointer hover:bg-green-100`}
          onClick={() => setSelectedNode(node.node_id)}
          style={{ marginLeft: `${depth * 20}px` }}
        >
          <div className="font-semibold text-sm">Leaf Node</div>
          <div className="text-xs text-muted-foreground">
            Class: {node.class === 0 ? 'Good Standing' : node.class === 1 ? 'Moderate Risk' : 'At-Risk'}
          </div>
          <div className="text-xs text-muted-foreground">Samples: {node.samples}</div>
          <div className="text-xs text-muted-foreground">Gini: {node.gini.toFixed(3)}</div>
        </div>
      );
    }

    return (
      <div key={node.node_id} style={{ marginLeft: `${depth * 20}px` }}>
        <div
          className={`p-2 rounded border mb-2 cursor-pointer ${
            selectedNode === node.node_id ? 'bg-blue-100 border-blue-500' : 'bg-yellow-50 border-yellow-300'
          } hover:bg-yellow-100`}
          onClick={() => setSelectedNode(node.node_id)}
        >
          <div className="font-semibold text-sm">{node.feature}</div>
          <div className="text-xs text-muted-foreground">Threshold: {node.threshold.toFixed(3)}</div>
          <div className="text-xs text-muted-foreground">Samples: {node.samples}</div>
          <div className="text-xs text-muted-foreground">Gini: {node.gini.toFixed(3)}</div>
        </div>
        <div className="flex flex-col gap-2">
          <div className="text-xs text-muted-foreground ml-2">← True</div>
          {treeStructure.find(n => n.node_id === node.left_child) && 
            renderTreeNode(treeStructure.find(n => n.node_id === node.left_child), depth + 1)}
          <div className="text-xs text-muted-foreground ml-2">→ False</div>
          {treeStructure.find(n => n.node_id === node.right_child) && 
            renderTreeNode(treeStructure.find(n => n.node_id === node.right_child), depth + 1)}
        </div>
      </div>
    );
  };

  const importanceData = Object.entries(featureImportance || {})
    .filter(([_, value]) => value > 0)
    .map(([feature, importance]) => ({ feature, importance: (importance * 100).toFixed(2) }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <TreeDeciduous className="w-4 h-4" />
            Decision Tree Structure
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-96 overflow-y-auto p-4 bg-white rounded border">
            {treeStructure && treeStructure.length > 0 ? (
              renderTreeNode(treeStructure[0])
            ) : (
              <p className="text-sm text-muted-foreground">No tree structure available</p>
            )}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="w-4 h-4" />
            Feature Importance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={importanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="feature" fontSize={10} angle={-45} textAnchor="end" height={80} />
              <YAxis fontSize={10} />
              <Tooltip />
              <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}

function RandomForestVisualization({ treeSamples, nTrees, featureImportance }) {
  const importanceData = Object.entries(featureImportance || {})
    .filter(([_, value]) => value > 0)
    .map(([feature, importance]) => ({ feature, importance: (importance * 100).toFixed(2) }))
    .sort((a, b) => b.importance - a.importance)
    .slice(0, 10);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Users className="w-4 h-4" />
            Random Forest Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="p-4 rounded bg-blue-50 border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{nTrees}</div>
              <div className="text-sm text-muted-foreground">Total Trees</div>
            </div>
            <div className="p-4 rounded bg-green-50 border border-green-200">
              <div className="text-2xl font-bold text-green-600">Majority Voting</div>
              <div className="text-sm text-muted-foreground">Aggregation Method</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Random Forest combines multiple decision trees through majority voting to improve prediction accuracy 
            and reduce overfitting. Each tree is trained on a random subset of data and features.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <TreeDeciduous className="w-4 h-4" />
            Sample Trees (First 3)
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            {treeSamples?.slice(0, 3).map((tree, idx) => (
              <div key={idx} className="p-4 rounded border">
                <div className="font-semibold mb-2">Tree {idx + 1}</div>
                <div className="text-xs text-muted-foreground">
                  {tree.filter(n => n.is_leaf).length} leaf nodes, {tree.filter(n => !n.is_leaf).length} internal nodes
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="w-4 h-4" />
            Feature Importance
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={importanceData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="feature" fontSize={10} angle={-45} textAnchor="end" height={80} />
              <YAxis fontSize={10} />
              <Tooltip />
              <Bar dataKey="importance" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}

function NaiveBayesVisualization({ classPriors, classCount, featureMeans, featureVar, featureNames }) {
  const classLabels = ['Good Standing', 'Moderate Risk', 'At-Risk'];
  
  const classData = classPriors?.map((prior, idx) => ({
    class: classLabels[idx] || `Class ${idx}`,
    prior: (prior * 100).toFixed(2),
    count: classCount?.[idx] || 0
  })) || [];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="w-4 h-4" />
            Class Priors
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={classData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="class" fontSize={10} />
              <YAxis fontSize={10} />
              <Tooltip />
              <Bar dataKey="prior" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} name="Prior Probability %" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Info className="w-4 h-4" />
            How Naive Bayes Works
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-3">
          <div className="p-3 rounded bg-blue-50 border border-blue-200">
            <div className="font-semibold text-sm mb-1">Formula</div>
            <div className="text-xs text-muted-foreground font-mono">
              P(Class | Features) ∝ P(Class) × P(Feature₁ | Class) × P(Feature₂ | Class) × ...
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Naive Bayes calculates the probability of each class given the input features using Bayes' theorem. 
            It assumes features are independent (naive assumption), making it fast and effective for high-dimensional data.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="w-4 h-4" />
            Feature Statistics
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="max-h-64 overflow-y-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2">Feature</th>
                  <th className="text-right p-2">Mean</th>
                  <th className="text-right p-2">Variance</th>
                </tr>
              </thead>
              <tbody>
                {featureNames?.slice(0, 10).map((feature, idx) => (
                  <tr key={idx} className="border-b">
                    <td className="p-2">{feature}</td>
                    <td className="text-right p-2">{featureMeans?.[0]?.[idx]?.toFixed(3) || 'N/A'}</td>
                    <td className="text-right p-2">{featureVar?.[0]?.[idx]?.toFixed(3) || 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function KNNVisualization({ nNeighbors, nSamplesFit, nFeatures }) {
  // Generate sample scatter plot data
  const scatterData = Array.from({ length: 50 }, (_, i) => ({
    x: Math.random() * 10,
    y: Math.random() * 10,
    class: i < 20 ? 'Good Standing' : i < 35 ? 'Moderate Risk' : 'At-Risk'
  }));

  const targetPoint = { x: 5, y: 5, class: 'Target' };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="w-4 h-4" />
            K-Nearest Neighbors Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <div className="p-4 rounded bg-blue-50 border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{nNeighbors}</div>
              <div className="text-sm text-muted-foreground">K Value</div>
            </div>
            <div className="p-4 rounded bg-green-50 border border-green-200">
              <div className="text-2xl font-bold text-green-600">{nSamplesFit}</div>
              <div className="text-sm text-muted-foreground">Training Samples</div>
            </div>
            <div className="p-4 rounded bg-purple-50 border border-purple-200">
              <div className="text-2xl font-bold text-purple-600">{nFeatures}</div>
              <div className="text-sm text-muted-foreground">Features</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            KNN classifies new data points based on the majority class of the K nearest neighbors in the feature space.
            Distance is typically measured using Euclidean distance.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Brain className="w-4 h-4" />
            2D Feature Space Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={350}>
            <ScatterChart>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" name="Feature 1" />
              <YAxis dataKey="y" name="Feature 2" />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name="Good Standing" data={scatterData.filter(d => d.class === 'Good Standing')} fill="green" />
              <Scatter name="Moderate Risk" data={scatterData.filter(d => d.class === 'Moderate Risk')} fill="yellow" />
              <Scatter name="At-Risk" data={scatterData.filter(d => d.class === 'At-Risk')} fill="red" />
              <Scatter name="Target" data={[targetPoint]} fill="blue" shape="star" />
            </ScatterChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-4 justify-center text-xs">
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-green-500" /> Good Standing</div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-yellow-500" /> Moderate Risk</div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-red-500" /> At-Risk</div>
            <div className="flex items-center gap-1"><div className="w-3 h-3 rounded-full bg-blue-500" /> Target</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function SVMVisualization({ nSupport, supportVectorsCount, dualCoef }) {
  // Generate sample data for visualization
  const lineData = Array.from({ length: 20 }, (_, i) => ({
    x: i,
    y: Math.sin(i * 0.5) * 2 + 5
  }));

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Zap className="w-4 h-4" />
            Support Vector Machine Overview
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4 mb-4">
            <div className="p-4 rounded bg-blue-50 border border-blue-200">
              <div className="text-2xl font-bold text-blue-600">{supportVectorsCount}</div>
              <div className="text-sm text-muted-foreground">Support Vectors</div>
            </div>
            <div className="p-4 rounded bg-green-50 border border-green-200">
              <div className="text-2xl font-bold text-green-600">Max Margin</div>
              <div className="text-sm text-muted-foreground">Optimization Goal</div>
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            SVM finds the optimal hyperplane that maximizes the margin between different classes. 
            Support vectors are the data points closest to the decision boundary.
          </p>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2">
            <Target className="w-4 h-4" />
            Decision Boundary Visualization
          </CardTitle>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={lineData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="x" fontSize={10} />
              <YAxis fontSize={10} />
              <Tooltip />
              <Line type="monotone" dataKey="y" stroke="hsl(var(--primary))" strokeWidth={2} name="Decision Boundary" />
              <Line type="monotone" dataKey="y" stroke="hsl(var(--destructive))" strokeWidth={1} strokeDasharray="5 5" name="Margin" />
            </LineChart>
          </ResponsiveContainer>
          <div className="flex items-center gap-4 mt-4 justify-center text-xs">
            <div className="flex items-center gap-1"><div className="w-8 h-1 bg-primary" /> Decision Boundary</div>
            <div className="flex items-center gap-1"><div className="w-8 h-1 bg-destructive border-dashed" /> Margin</div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

function AIExplanation({ algorithm, featureImportance, visualizationData }) {
  const topFeatures = Object.entries(featureImportance || {})
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const explanations = {
    'Decision Tree': 'Decision Tree makes predictions by recursively splitting the data based on feature values. Each split aims to maximize information gain and minimize impurity.',
    'Random Forest': 'Random Forest combines multiple decision trees through majority voting. Each tree is trained on a random subset of data and features, reducing overfitting.',
    'Naive Bayes': 'Naive Bayes uses Bayes\' theorem with the assumption that features are independent. It calculates the probability of each class given the input features.',
    'KNN': 'K-Nearest Neighbors classifies data based on the majority class of the K nearest neighbors in the feature space. It\'s a simple, instance-based learning algorithm.',
    'SVM': 'Support Vector Machine finds the optimal hyperplane that maximizes the margin between different classes. It\'s effective for high-dimensional data.'
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base flex items-center gap-2">
          <Info className="w-4 h-4" />
          AI Explanation
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="p-4 rounded bg-blue-50 border border-blue-200">
          <div className="font-semibold text-sm mb-2">How {algorithm} Works</div>
          <p className="text-sm text-muted-foreground">{explanations[algorithm] || 'Algorithm explanation not available.'}</p>
        </div>

        <div>
          <div className="font-semibold text-sm mb-3">Top Contributing Features</div>
          <div className="space-y-2">
            {topFeatures.map(([feature, importance], idx) => (
              <div key={idx} className="flex items-center gap-3 p-2 rounded bg-muted/50">
                <div className="w-6 h-6 rounded-full bg-primary text-white flex items-center justify-center text-xs font-bold">
                  {idx + 1}
                </div>
                <div className="flex-1">
                  <div className="text-sm font-medium">{feature}</div>
                  <div className="w-full h-2 bg-gray-200 rounded-full mt-1">
                    <div 
                      className="h-full bg-primary rounded-full" 
                      style={{ width: `${(importance * 100).toFixed(1)}%` }}
                    />
                  </div>
                </div>
                <div className="text-sm font-bold text-primary">{(importance * 100).toFixed(1)}%</div>
              </div>
            ))}
          </div>
        </div>

        <div className="p-4 rounded bg-green-50 border border-green-200">
          <div className="font-semibold text-sm mb-2">Interpretation</div>
          <p className="text-sm text-muted-foreground">
            The model primarily considers {topFeatures[0]?.[0] || 'key features'} when making predictions. 
            Students with lower values in these features are more likely to be classified as at-risk.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}

export default function AlgorithmVisualization({ algorithm, sessionData }) {
  const [isLoading, setIsLoading] = useState(false);

  const { data: vizData, isLoading: vizLoading, error: vizError } = useQuery({
    queryKey: ['model-visualization', algorithm],
    queryFn: async () => {
      const response = await fetch('http://localhost:5000/api/model-visualization', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ algorithm })
      });
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || 'Failed to fetch visualization data');
      }
      return response.json();
    },
    enabled: !!algorithm,
  });

  if (vizLoading) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4" />
            <p className="text-sm text-muted-foreground">Loading visualization data...</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  if (vizError) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center py-12">
          <div className="text-center">
            <p className="text-sm text-red-600 mb-2">Error loading visualization</p>
            <p className="text-xs text-muted-foreground">{vizError.message}</p>
          </div>
        </CardContent>
      </Card>
    );
  }

  const visualizationData = vizData?.visualization_data;

  return (
    <div className="space-y-4">
      <div>
        <h2 className="text-lg font-bold">Algorithm Interpretation & Training Visualization</h2>
        <p className="text-sm text-muted-foreground">Understanding how {algorithm} makes predictions</p>
      </div>

      <div className="space-y-4">
        {algorithm === 'Decision Tree' && visualizationData?.tree_structure && (
          <DecisionTreeVisualization 
            treeStructure={visualizationData.tree_structure}
            featureImportance={visualizationData.feature_importance}
          />
        )}
        {algorithm === 'Random Forest' && visualizationData?.tree_samples && (
          <RandomForestVisualization 
            treeSamples={visualizationData.tree_samples}
            nTrees={visualizationData.n_trees}
            featureImportance={visualizationData.feature_importance}
          />
        )}
        {algorithm === 'Naive Bayes' && visualizationData?.class_priors && (
          <NaiveBayesVisualization 
            classPriors={visualizationData.class_priors}
            classCount={visualizationData.class_count}
            featureMeans={visualizationData.feature_means}
            featureVar={visualizationData.feature_var}
            featureNames={visualizationData.feature_names}
          />
        )}
        {algorithm === 'KNN' && visualizationData?.n_neighbors && (
          <KNNVisualization 
            nNeighbors={visualizationData.n_neighbors}
            nSamplesFit={visualizationData.n_samples_fit}
            nFeatures={visualizationData.n_features}
          />
        )}
        {algorithm === 'SVM' && visualizationData?.support_vectors_count && (
          <SVMVisualization 
            nSupport={visualizationData.n_support}
            supportVectorsCount={visualizationData.support_vectors_count}
            dualCoef={visualizationData.dual_coef}
          />
        )}
        {!visualizationData && (
          <Card>
            <CardContent className="flex items-center justify-center py-12">
              <p className="text-sm text-muted-foreground">No visualization data available for this algorithm</p>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
