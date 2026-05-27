import React, { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { supabase as base44 } from '@/api/supabaseClient';
import PageHeader from '@/components/shared/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, LineChart, Line } from 'recharts';
import { Info, BarChart3, TrendingUp, Lightbulb, Zap, Target, Brain } from 'lucide-react';
import AlgorithmVisualization from '@/components/admin/AlgorithmVisualization';

const METRIC_EXPLANATIONS = {
  accuracy: {
    title: 'Accuracy',
    explanation: 'Ratio of correctly predicted observations to total observations. Calculated as (TP + TN) / (TP + TN + FP + FN). Measures overall model correctness across all classes.',
    interpretation: 'Higher values indicate better overall classification performance. Values near 1.0 suggest the model correctly classifies most instances.'
  },
  precision: {
    title: 'Precision',
    explanation: 'Ratio of correctly predicted positive observations to total predicted positive observations. Calculated as TP / (TP + FP). Measures how many selected items are relevant.',
    interpretation: 'Higher values indicate fewer false positives. Critical when minimizing false alarms is important (e.g., avoiding mislabeling low-risk students as high-risk).'
  },
  recall: {
    title: 'Recall (Sensitivity)',
    explanation: 'Ratio of correctly predicted positive observations to all actual positive observations. Calculated as TP / (TP + FN). Measures ability to find all positive samples.',
    interpretation: 'Higher values indicate fewer false negatives. Critical when missing positive cases is costly (e.g., failing to identify at-risk students).'
  },
  f1_score: {
    title: 'F1 Score',
    explanation: 'Harmonic mean of Precision and Recall. Calculated as 2 × (Precision × Recall) / (Precision + Recall). Balances both metrics when class distribution is imbalanced.',
    interpretation: 'Higher values indicate better balance between precision and recall. Useful when you need to consider both false positives and false negatives equally.'
  },
  roc_auc: {
    title: 'ROC AUC',
    explanation: 'Area Under the Receiver Operating Characteristic Curve. Measures the model\'s ability to distinguish between classes across different threshold settings. Based on True Positive Rate vs False Positive Rate.',
    interpretation: 'Values range from 0.5 (random guessing) to 1.0 (perfect classification). Higher values indicate better discriminative ability across all classification thresholds.'
  }
};

export default function ModelResults() {
  const [activeTab, setActiveTab] = useState('overview');
  const { data: logs = [] } = useQuery({
    queryKey: ['trainingLogs'],
    queryFn: () => base44.entities.TrainingLog.list(),
    refetchInterval: 5000
  });

  const sortedLogs = [...logs].sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
  const latestSession = sortedLogs.length > 0 ? sortedLogs[0]?.training_session_id : null;
  const sessionLogs = sortedLogs.filter(l => l.training_session_id === latestSession);
  const bestModel = sessionLogs.find(l => l.is_best);

  const accuracyData = sessionLogs.map(l => ({
    name: l.algorithm,
    accuracy: +(l.accuracy * 100).toFixed(1),
    precision: +(l.precision * 100).toFixed(1),
    recall: +(l.recall * 100).toFixed(1),
    f1: +(l.f1_score * 100).toFixed(1),
    roc_auc: +(l.roc_auc * 100).toFixed(1),
  }));

  const radarData = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].map(metric => {
    const point = { metric: metric.toUpperCase().replace('_', ' ') };
    sessionLogs.forEach(l => {
      point[l.algorithm] = +(l[metric === 'f1' ? 'f1_score' : metric] * 100).toFixed(1);
    });
    return point;
  });

  if (sessionLogs.length === 0) {
    return (
      <div>
        <PageHeader title="Model Results" />
        <Card className="max-w-lg mx-auto mt-12">
          <CardContent className="py-12 text-center text-muted-foreground">
            No training results yet. Train the model first.
          </CardContent>
        </Card>
      </div>
    );
  }

  return (
    <div>
      <PageHeader title="Model Results & Visualization" description={`Latest training session • Best: ${bestModel?.algorithm}`} />

      {/* Tab Navigation */}
      <div className="flex gap-2 flex-wrap mb-6">
        <Button
          variant={activeTab === 'overview' ? 'default' : 'outline'}
          onClick={() => setActiveTab('overview')}
          className="flex items-center gap-2"
        >
          <BarChart3 className="w-4 h-4" />
          Overview
        </Button>
        <Button
          variant={activeTab === 'metrics' ? 'default' : 'outline'}
          onClick={() => setActiveTab('metrics')}
          className="flex items-center gap-2"
        >
          <TrendingUp className="w-4 h-4" />
          Metrics Analysis
        </Button>
        <Button
          variant={activeTab === 'insights' ? 'default' : 'outline'}
          onClick={() => setActiveTab('insights')}
          className="flex items-center gap-2"
        >
          <Lightbulb className="w-4 h-4" />
          Model Insights
        </Button>
        <Button
          variant={activeTab === 'recommendations' ? 'default' : 'outline'}
          onClick={() => setActiveTab('recommendations')}
          className="flex items-center gap-2"
        >
          <Zap className="w-4 h-4" />
          Recommendations
        </Button>
        <Button
          variant={activeTab === 'algorithm' ? 'default' : 'outline'}
          onClick={() => setActiveTab('algorithm')}
          className="flex items-center gap-2"
        >
          <Brain className="w-4 h-4" />
          Algorithm Visualization
        </Button>
      </div>

      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
        <Card>
          <CardHeader>
            <CardTitle className="text-base">Accuracy Comparison</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" fontSize={11} />
                <YAxis domain={[0, 100]} fontSize={11} />
                <Tooltip contentStyle={{ borderRadius: '8px', border: '1px solid hsl(var(--border))' }} />
                <Bar dataKey="accuracy" fill="hsl(var(--primary))" radius={[4, 4, 0, 0]} name="Accuracy %" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle className="text-base">Metrics Radar</CardTitle>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="hsl(var(--border))" />
                <PolarAngleAxis dataKey="metric" fontSize={10} />
                <PolarRadiusAxis domain={[0, 100]} fontSize={10} />
                {sessionLogs.map((l, i) => (
                  <Radar key={l.algorithm} name={l.algorithm} dataKey={l.algorithm}
                    stroke={`hsl(var(--chart-${(i % 5) + 1}))`}
                    fill={`hsl(var(--chart-${(i % 5) + 1}))`}
                    fillOpacity={0.1} />
                ))}
                <Legend />
              </RadarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader>
          <CardTitle className="text-base">Metrics Table</CardTitle>
        </CardHeader>
        <CardContent>
          <Table>
            <TableHeader>
              <TableRow className="bg-muted/50">
                <TableHead>Algorithm</TableHead>
                <TableHead>Accuracy</TableHead>
                <TableHead>Precision</TableHead>
                <TableHead>Recall</TableHead>
                <TableHead>F1 Score</TableHead>
                <TableHead>ROC AUC</TableHead>
                <TableHead>Status</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sessionLogs.map(l => (
                <TableRow key={l.id}>
                  <TableCell className="font-medium">{l.algorithm}</TableCell>
                  <TableCell>{(l.accuracy * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(l.precision * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(l.recall * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(l.f1_score * 100).toFixed(1)}%</TableCell>
                  <TableCell>{(l.roc_auc * 100).toFixed(1)}%</TableCell>
                  <TableCell>
                    {l.is_best ? (
                      <Badge className="bg-emerald-100 text-emerald-700 border-emerald-200">Best Model</Badge>
                    ) : (
                      <Badge variant="outline">Evaluated</Badge>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-base">Metric Explanations</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          {Object.entries(METRIC_EXPLANATIONS).map(([key, info]) => (
            <div key={key} className="p-4 rounded-lg bg-muted/50 space-y-2">
              <div className="flex items-center gap-2">
                <Info className="w-4 h-4 text-primary" />
                <h4 className="font-semibold text-sm">{info.title}</h4>
              </div>
              <p className="text-xs text-muted-foreground">{info.explanation}</p>
              <p className="text-xs text-muted-foreground italic">{info.interpretation}</p>
            </div>
          ))}
        </CardContent>
      </Card>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
        <Card>
          <CardHeader><CardTitle className="text-base">All Metrics Comparison</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" fontSize={10} />
                <YAxis domain={[0, 100]} fontSize={10} />
                <Tooltip contentStyle={{ borderRadius: '8px' }} />
                <Legend />
                <Bar dataKey="accuracy" fill="hsl(var(--chart-1))" radius={[2,2,0,0]} />
                <Bar dataKey="precision" fill="hsl(var(--chart-2))" radius={[2,2,0,0]} />
                <Bar dataKey="recall" fill="hsl(var(--chart-3))" radius={[2,2,0,0]} />
                <Bar dataKey="f1" fill="hsl(var(--chart-4))" radius={[2,2,0,0]} name="F1 Score" />
                <Bar dataKey="roc_auc" fill="hsl(var(--chart-5))" radius={[2,2,0,0]} name="ROC AUC" />
              </BarChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        <Card>
          <CardHeader><CardTitle className="text-base">ROC-AUC & F1 Score Comparison</CardTitle></CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={280}>
              <LineChart data={accuracyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" fontSize={10} />
                <YAxis domain={[0, 100]} fontSize={10} />
                <Tooltip contentStyle={{ borderRadius: '8px' }} />
                <Legend />
                <Line type="monotone" dataKey="roc_auc" stroke="hsl(var(--chart-5))" strokeWidth={2} dot={{ r: 5 }} name="ROC AUC" />
                <Line type="monotone" dataKey="f1" stroke="hsl(var(--chart-4))" strokeWidth={2} dot={{ r: 5 }} name="F1 Score" />
                <Line type="monotone" dataKey="precision" stroke="hsl(var(--chart-2))" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} name="Precision" />
                <Line type="monotone" dataKey="recall" stroke="hsl(var(--chart-3))" strokeWidth={2} strokeDasharray="5 5" dot={{ r: 4 }} name="Recall" />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>
        </div>
      )}

      {activeTab === 'metrics' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <TrendingUp className="w-4 h-4" />
              Detailed Metrics Breakdown
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {sessionLogs.map((result, index) => (
                <div key={index} className={`p-4 rounded-lg border ${result.algorithm === bestModel?.algorithm ? 'bg-emerald-50 border-emerald-200' : 'bg-muted/50'}`}>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-2">
                      <span className="font-medium">{result.algorithm}</span>
                      {result.algorithm === bestModel?.algorithm && (
                        <Badge variant="default" className="bg-emerald-500">Best Model</Badge>
                      )}
                    </div>
                    <span className="text-sm font-semibold text-emerald-600">
                      {(result.accuracy * 100).toFixed(1)}% Accuracy
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                    <div>
                      <p className="text-muted-foreground">Precision</p>
                      <p className="font-medium">{(result.precision * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">Recall</p>
                      <p className="font-medium">{(result.recall * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">F1 Score</p>
                      <p className="font-medium">{(result.f1_score * 100).toFixed(1)}%</p>
                    </div>
                    <div>
                      <p className="text-muted-foreground">ROC AUC</p>
                      <p className="font-medium">{((result.roc_auc || 0) * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === 'insights' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Lightbulb className="w-4 h-4" />
              Model Interpretation & Insights
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">Why {bestModel?.algorithm} Performed Best</h4>
              <p className="text-sm text-blue-700">
                {bestModel?.algorithm === 'Random Forest' && 
                  "Random Forest excels by combining multiple decision trees, reducing overfitting and capturing complex non-linear relationships in student data. It handles mixed feature types well and provides robust predictions even with noisy data."}
                {bestModel?.algorithm === 'Decision Tree' && 
                  "Decision Tree performed well due to its ability to create clear, interpretable rules for student classification. It handles both numerical and categorical features effectively without requiring extensive data preprocessing."}
                {bestModel?.algorithm === 'SVM' && 
                  "SVM achieved high accuracy by finding optimal hyperplanes that separate student classes. It works particularly well with the current feature space and handles the margin between classes effectively."}
                {bestModel?.algorithm === 'KNN' && 
                  "KNN performed well because similar students tend to have similar academic outcomes. The local decision-making approach captures patterns that global models might miss."}
                {bestModel?.algorithm === 'Naive Bayes' && 
                  "Naive Bayes showed strong performance due to the independence assumptions holding reasonably well for the student features. It's particularly effective with the current dataset size and feature distribution."}
              </p>
            </div>

            <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
              <h4 className="font-semibold text-purple-800 mb-2">Data Processing Insights</h4>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• Dataset size: {sessionLogs[0]?.dataset_size || 'N/A'} students processed</li>
                <li>• Features used: GPA history, financial status, personal factors, academic performance</li>
                <li>• Training split: 80% training, 20% testing</li>
                <li>• Cross-validation: 5-fold validation applied</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg bg-amber-50 border border-amber-200">
              <h4 className="font-semibold text-amber-800 mb-2">Algorithm Strengths Analysis</h4>
              <div className="text-sm text-amber-700 space-y-2">
                {sessionLogs.map((r, i) => (
                  <div key={i} className="flex justify-between items-center">
                    <span>{r.algorithm}</span>
                    <div className="flex items-center gap-2">
                      <div className="w-24 h-2 bg-amber-200 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-amber-600" 
                          style={{ width: `${(r.accuracy * 100)}%` }}
                        />
                      </div>
                      <span className="text-xs font-medium">{(r.accuracy * 100).toFixed(1)}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === 'recommendations' && (
        <Card>
          <CardHeader>
            <CardTitle className="text-base flex items-center gap-2">
              <Zap className="w-4 h-4" />
              Recommendations to Improve Algorithm Performance
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-lg bg-emerald-50 border border-emerald-200">
              <h4 className="font-semibold text-emerald-800 mb-2">Data Quality Improvements</h4>
              <ul className="text-sm text-emerald-700 space-y-1">
                <li>• Increase dataset size - more student records improve model generalization</li>
                <li>• Add more features - attendance records, study hours, extracurricular activities</li>
                <li>• Handle missing values - implement imputation strategies for incomplete GPA data</li>
                <li>• Feature engineering - create composite features like GPA trend, semester consistency</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg bg-blue-50 border border-blue-200">
              <h4 className="font-semibold text-blue-800 mb-2">Algorithm Tuning Options</h4>
              <ul className="text-sm text-blue-700 space-y-1">
                <li>• Hyperparameter optimization - use GridSearch or RandomSearch for better parameters</li>
                <li>• Ensemble methods - combine multiple algorithms for improved accuracy</li>
                <li>• Feature selection - identify and use only the most predictive features</li>
                <li>• Class balancing - address imbalance between Good Standing and At-Risk students</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg bg-purple-50 border border-purple-200">
              <h4 className="font-semibold text-purple-800 mb-2">Model-Specific Recommendations</h4>
              <ul className="text-sm text-purple-700 space-y-1">
                <li>• <strong>Random Forest:</strong> Increase number of trees, adjust max depth</li>
                <li>• <strong>SVM:</strong> Try different kernels (RBF, polynomial), adjust C and gamma parameters</li>
                <li>• <strong>KNN:</strong> Optimize k value, implement weighted voting</li>
                <li>• <strong>Decision Tree:</strong> Prune trees to prevent overfitting, adjust min samples split</li>
                <li>• <strong>Naive Bayes:</strong> Try different feature distributions, handle correlated features</li>
              </ul>
            </div>

            <div className="p-4 rounded-lg bg-red-50 border border-red-200">
              <h4 className="font-semibold text-red-800 mb-2">Validation & Monitoring</h4>
              <ul className="text-sm text-red-700 space-y-1">
                <li>• Implement cross-validation with more folds (10-fold)</li>
                <li>• Monitor model drift over time as student patterns change</li>
                <li>• A/B test different algorithms in production</li>
                <li>• Collect feedback from deans on prediction accuracy</li>
              </ul>
            </div>
          </CardContent>
        </Card>
      )}

      {activeTab === 'algorithm' && (
        <AlgorithmVisualization 
          algorithm={bestModel?.algorithm || 'Decision Tree'}
          sessionData={bestModel}
        />
      )}
    </div>
  );
}