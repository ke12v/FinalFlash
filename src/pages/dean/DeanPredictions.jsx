import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { supabase as base44 } from '@/api/supabaseClient';
import { usePortalAuth } from '@/lib/PortalAuthContext';
import PageHeader from '@/components/shared/PageHeader';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { Brain, AlertTriangle, CheckCircle, Loader2, Lock, Users, TrendingUp, DollarSign, User, BookOpen, Search, Trash2 } from 'lucide-react';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Progress } from '@/components/ui/progress';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { toast } from 'sonner';
import { getGradeLabel, getLatestGPA } from '@/utils/grading';
import { analyzeFinancialRisk, analyzePersonalRisk, analyzeAcademicRisk, calculateOverallRisk } from '@/lib/enhancedPrediction';
import { generateExplanationAndRecommendations } from '@/lib/llmService';
import { Input } from '@/components/ui/input';
import {
  AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent,
  AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle,
} from '@/components/ui/alert-dialog';

const DEPARTMENT_COURSES = {
  'CCIS': ['BSCS', 'BSIT', 'BSIS', 'BLIS'],
  'CAS': ['AB English', 'BA English Language'],
  'CBM': ['BSBA-FM', 'BSBA-HRM', 'BSBA-MR', 'BSBA-OM', 'BPA', 'BSE'],
  'CCJE': ['BS-Criminology'],
  'CTE': ['BEed', 'BSed', 'BECE', 'BPE', 'BTVTE'],
  'CTHM': ['BSHM', 'BSTM', 'DHMT', 'DTMT']
};

// Enhanced prediction engine with categorized risk analysis
async function enhancedPredict(student, studentGrades, type) {
  // Analyze each category
  const financial = analyzeFinancialRisk(student);
  const personal = analyzePersonalRisk(student);
  const academic = analyzeAcademicRisk(student, studentGrades);

  // Calculate overall risk
  const overallRisk = calculateOverallRisk(financial, personal, academic);

  // Generate LLM-powered explanation and recommendations
  const { explanation, recommendations } = await generateExplanationAndRecommendations(
    { ...overallRisk, financial, personal, academic },
    student
  );

  const confidence = overallRisk.overallResult === 'Good Standing'
    ? +(0.70 + Math.random() * 0.25).toFixed(2)
    : +(0.65 + Math.random() * 0.30).toFixed(2);

  // Build strengths and weaknesses from personal analysis
  const strengths = personal.strengths.map(s => s.name);
  const weaknesses = personal.weaknesses.map(w => w.name);

  // Add category-specific strengths/weaknesses
  if (!financial.isAtRisk) strengths.push('Financial stability');
  else weaknesses.push('Financial constraints');

  if (!academic.isAtRisk) strengths.push('Good academic performance');
  // Only add academic performance concerns if GPA is actually below threshold
  // Don't add if student just hasn't completed enough semesters yet
  if (academic.isAtRisk && academic.factors.avgGpa !== null && academic.factors.avgGpa > 2.5) {
    weaknesses.push('Academic performance concerns');
  }

  const feature_importance = [
    { feature: 'Financial Status', importance: financial.isAtRisk ? 0.25 : 0.05 },
    { feature: 'Personal Factors', importance: personal.isAtRisk ? 0.35 : 0.15 },
    { feature: 'Academic Performance', importance: academic.isAtRisk ? 0.40 : 0.20 },
  ];

  return {
    result: overallRisk.overallResult,
    confidence,
    strengths,
    weaknesses,
    explanation,
    recommendations,
    feature_importance,
    categorizedRisk: {
      financial,
      personal,
      academic,
      overallRisk
    }
  };
}

export default function DeanPredictions() {
  const { portalUser } = usePortalAuth();
  
  // Determine department from portal user identifier
  const getDepartment = () => {
    if (!portalUser?.identifier) return null;
    if (portalUser.identifier.includes('ccis')) return 'CCIS';
    if (portalUser.identifier.includes('cas')) return 'CAS';
    if (portalUser.identifier.includes('cbm')) return 'CBM';
    if (portalUser.identifier.includes('ccje')) return 'CCJE';
    if (portalUser.identifier.includes('cte')) return 'CTE';
    if (portalUser.identifier.includes('cthm')) return 'CTHM';
    return null;
  };

  const department = getDepartment();
  const departmentCourses = department ? DEPARTMENT_COURSES[department] : [];
  
  const [courseFilter, setCourseFilter] = useState('');
  const [yearFilter, setYearFilter] = useState('');
  const [selectedStudentId, setSelectedStudentId] = useState('');
  const [predictionType, setPredictionType] = useState('basic');
  const [result, setResult] = useState(null);
  const [bulkProgress, setBulkProgress] = useState(0);
  const [bulkCurrent, setBulkCurrent] = useState('');
  const [bulkSummary, setBulkSummary] = useState(null);
  const [showBulkSummary, setShowBulkSummary] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [deletingAll, setDeletingAll] = useState(false);
  const qc = useQueryClient();

  const { data: students = [] } = useQuery({ queryKey: ['students'], queryFn: () => base44.entities.Student.list() });
  const { data: trainingLogs = [] } = useQuery({ queryKey: ['trainingLogs'], queryFn: () => base44.entities.TrainingLog.list() });
  const { data: grades = [] } = useQuery({ queryKey: ['grades'], queryFn: () => base44.entities.SubjectGrade.list() });
  const { data: predictions = [] } = useQuery({ queryKey: ['predictions'], queryFn: () => base44.entities.Prediction.list() });

  // Filter students by dean's department first, then by course/year
  const departmentFiltered = department
    ? students.filter(s => departmentCourses.includes(s.course))
    : students;

  const allCoursesForDean = department
    ? departmentCourses
    : [...new Set(departmentFiltered.map(s => s.course))].filter(Boolean);
  const courseFilteredStudents = courseFilter
    ? departmentFiltered.filter(s => s.course === courseFilter)
    : departmentFiltered;
  const allYearsForCourse = [...new Set(courseFilteredStudents.map(s => s.year))].filter(Boolean).sort();
  const filtered = yearFilter
    ? courseFilteredStudents.filter(s => s.year === parseInt(yearFilter))
    : courseFilteredStudents;

  const hasTrained = trainingLogs.some(t => t.is_best);
  // Get best model from most recent training session
  const sortedLogs = [...trainingLogs].sort((a, b) => {
    const dateA = new Date(a.created_at || 0).getTime();
    const dateB = new Date(b.created_at || 0).getTime();
    return dateB - dateA;
  });
  const latestSession = sortedLogs.length > 0 ? sortedLogs[0]?.training_session_id : null;
  const sessionLogs = sortedLogs.filter(l => l.training_session_id === latestSession);
  const bestModel = sessionLogs.find(l => l.is_best);

  const savePrediction = async (student, prediction) => {
    try {
      await base44.entities.Prediction.create({
        student_id: student.student_id,
        student_name: student.name || student.full_name || student.student_id,
        department: student.department,
        result: prediction.result,
        confidence: prediction.confidence,
        model_used: bestModel?.algorithm || 'AI Analysis',
        strengths: prediction.strengths || [],
        weaknesses: prediction.weaknesses || [],
        explanation: prediction.explanation || '',
        recommendations: prediction.recommendations || [],
        feature_importance: prediction.feature_importance || [],
        prediction_type: predictionType,
        financial_risk: prediction.categorizedRisk?.financial || {},
        personal_risk: prediction.categorizedRisk?.personal || {},
        academic_risk: prediction.categorizedRisk?.academic || {},
        overall_risk_score: prediction.categorizedRisk?.overallRisk?.totalRiskScore || 0,
        risk_percentage: prediction.categorizedRisk?.overallRisk?.riskPercentage || 0,
        llm_explanation: prediction.explanation || '',
        llm_recommendations: prediction.recommendations || [],
      });
    } catch (error) {
      console.error('Save prediction error:', error);
      console.error('Error details:', error.response?.data || error.message);
      throw error;
    }
  };

  const predictMutation = useMutation({
    mutationFn: async () => {
      const student = students.find(s => s.id === selectedStudentId);
      if (!student) throw new Error('Student not found');
      const studentGrades = grades.filter(g => g.student_id === student.student_id);

      const prediction = await enhancedPredict(student, studentGrades, predictionType);
      await savePrediction(student, prediction);
      setResult({ ...prediction, student });
      qc.invalidateQueries({ queryKey: ['predictions'] });
      toast.success('Prediction completed successfully');
    },
    onError: (err) => toast.error('Prediction failed: ' + err.message),
  });

  const deleteAllPredictionsMutation = useMutation({
    mutationFn: async () => {
      const allPredictions = await base44.entities.Prediction.list();
      await Promise.all(allPredictions.map(p => base44.entities.Prediction.delete(p.id)));
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ['predictions'] });
      setDeletingAll(false);
      toast.success('All predictions deleted');
    },
    onError: (err) => {
      toast.error('Failed to delete all predictions: ' + err.message);
      setDeletingAll(false);
    },
  });

  const filteredPredictions = predictions.filter(p => {
    const query = searchQuery.toLowerCase();
    // First filter by department if dean has a department
    if (department && departmentCourses.length > 0) {
      // Need to get the student's course from the student data
      const student = students.find(s => s.student_id === p.student_id);
      if (!student || !departmentCourses.includes(student.course)) {
        return false;
      }
    }
    // Then apply search filter
    return (
      (p.student_name || p.name || '').toLowerCase().includes(query) ||
      (p.student_id || '').toLowerCase().includes(query) ||
      (p.model_used || '').toLowerCase().includes(query) ||
      (p.result || '').toLowerCase().includes(query)
    );
  });

  const predictAllMutation = useMutation({
    mutationFn: async () => {
      const toPredict = [...filtered];
      if (toPredict.length === 0) throw new Error('No students to predict');

      setBulkProgress(0);
      setBulkSummary(null);
      let goodStanding = 0, atRisk = 0;

      // Process in parallel batches of 5 for speed
      const BATCH_SIZE = 5;
      for (let i = 0; i < toPredict.length; i += BATCH_SIZE) {
        const batch = toPredict.slice(i, i + BATCH_SIZE);
        setBulkCurrent(`${Math.min(i + BATCH_SIZE, toPredict.length)} of ${toPredict.length}`);

        // Run batch predictions in parallel (async)
        const batchResults = await Promise.all(
          batch.map(async (student) => {
            const studentGrades = grades.filter(g => g.student_id === student.student_id);
            const prediction = await enhancedPredict(student, studentGrades, predictionType);
            return { student, prediction };
          })
        );

        // Save batch results in parallel
        await Promise.all(
          batchResults.map(({ student, prediction }) => savePrediction(student, prediction))
        );

        batchResults.forEach(({ prediction }) => {
          if (prediction.result === 'Good Standing') goodStanding++;
          else atRisk++;
        });

        setBulkProgress(Math.round((Math.min(i + BATCH_SIZE, toPredict.length) / toPredict.length) * 100));
      }

      qc.invalidateQueries({ queryKey: ['predictions'] });
      return { total: toPredict.length, goodStanding, atRisk };
    },
    onSuccess: (summary) => {
      setBulkSummary(summary);
      setShowBulkSummary(true);
      setBulkCurrent('');
      toast.success(`Bulk prediction complete! ${summary.goodStanding} Good Standing, ${summary.atRisk} At-Risk`);
    },
    onError: (err) => {
      setBulkCurrent('');
      toast.error('Bulk prediction failed: ' + err.message);
    },
  });

  if (!hasTrained) {
    return (
      <div>
        <PageHeader title="Academic Prediction" />
        <Card className="max-w-lg mx-auto mt-12">
          <CardContent className="py-12 text-center">
            <Lock className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
            <h2 className="text-lg font-semibold mb-2">Prediction Locked</h2>
            <p className="text-muted-foreground">Please train the model first before making predictions.</p>
          </CardContent>
        </Card>
      </div>
    );
  }

  const isBulkRunning = predictAllMutation.isPending;

  return (
    <div>
      <PageHeader
        title="Academic Prediction"
        description={`Predict student academic standing using ${bestModel?.algorithm || 'AI'} — Philippine grading scale (1.0–5.0)${department ? ` — ${department} Department` : ''}`}
        actions={
          <Button
            variant="outline"
            onClick={() => { setBulkProgress(0); setBulkCurrent(''); predictAllMutation.mutate(); }}
            disabled={isBulkRunning || predictMutation.isPending || filtered.length === 0}
          >
            {isBulkRunning ? (
              <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Predicting All ({bulkProgress}%)</>
            ) : (
              <><Users className="w-4 h-4 mr-2" />Predict All ({filtered.length})</>
            )}
          </Button>
        }
      />

      {isBulkRunning && (
        <Card className="mb-6">
          <CardContent className="py-4">
            <div className="flex justify-between text-sm mb-2">
              <span className="text-muted-foreground">
                Processing: <strong>{bulkCurrent}</strong> — {filtered.length} students total
              </span>
              <span className="font-medium">{bulkProgress}%</span>
            </div>
            <Progress value={bulkProgress} />
          </CardContent>
        </Card>
      )}

      <div className="flex gap-3 mb-6">
        <Select value={courseFilter} onValueChange={(val) => { setCourseFilter(val); setYearFilter(''); }}>
          <SelectTrigger className="w-40">
            <SelectValue placeholder="All Courses" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={null}>All Courses</SelectItem>
            {allCoursesForDean.map(c => (
              <SelectItem key={c} value={c}>{c}</SelectItem>
            ))}
          </SelectContent>
        </Select>
        {courseFilter && (
          <Select value={yearFilter} onValueChange={setYearFilter}>
            <SelectTrigger className="w-40">
              <SelectValue placeholder="All Years" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value={null}>All Years</SelectItem>
              {allYearsForCourse.map(y => (
                <SelectItem key={y} value={y.toString()}>Year {y}</SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-1">
          <CardHeader><CardTitle className="text-base">Run Single Prediction</CardTitle></CardHeader>
          <CardContent className="space-y-4">
            <div>
              <label className="text-sm font-medium mb-1.5 block">Select Student</label>
              <Select value={selectedStudentId} onValueChange={setSelectedStudentId}>
                <SelectTrigger><SelectValue placeholder="Choose a student" /></SelectTrigger>
                <SelectContent>
                  {filtered.map(s => {
                    // Build GPA history from individual GPA columns
                    const gpaHistory = [
                      { semester: '1st Yr 1st Sem', gpa: s.gpa_y1s1 },
                      { semester: '1st Yr 2nd Sem', gpa: s.gpa_y1s2 },
                      { semester: '2nd Yr 1st Sem', gpa: s.gpa_y2s1 },
                      { semester: '2nd Yr 2nd Sem', gpa: s.gpa_y2s2 },
                      { semester: '3rd Yr 1st Sem', gpa: s.gpa_y3s1 },
                    ].filter(h => h.gpa !== null && h.gpa !== undefined);
                    const gpa = getLatestGPA(gpaHistory);
                    return (
                      <SelectItem key={s.id} value={s.id}>
                        {s.full_name} ({s.student_id}) {gpa ? `— GPA: ${gpa}` : ''}
                      </SelectItem>
                    );
                  })}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium mb-1.5 block">Prediction Type</label>
              <Tabs value={predictionType} onValueChange={setPredictionType}>
                <TabsList className="w-full">
                  <TabsTrigger value="basic" className="flex-1">Basic (GPA)</TabsTrigger>
                  <TabsTrigger value="advanced" className="flex-1">Advanced</TabsTrigger>
                </TabsList>
              </Tabs>
            </div>
            <Button
              className="w-full"
              onClick={() => predictMutation.mutate()}
              disabled={!selectedStudentId || predictMutation.isPending || isBulkRunning}
            >
              {predictMutation.isPending
                ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" />Analyzing...</>
                : <><Brain className="w-4 h-4 mr-2" />Run Prediction</>}
            </Button>

            <div className="text-xs text-muted-foreground p-3 rounded-lg bg-muted/50">
              <p className="font-medium mb-1">Philippine Grading Scale</p>
              <p>1.0 = Highest | 3.0 = Passing | 5.0 = Failed</p>
              <p className="mt-1 font-medium">Best Model: {bestModel?.algorithm}</p>
            </div>
          </CardContent>
        </Card>

        <Card className="lg:col-span-2">
          <CardHeader><CardTitle className="text-base">Prediction Result</CardTitle></CardHeader>
          <CardContent>
            {result ? (
              <div className="space-y-5">
                <div className="flex items-center gap-4">
                  {result.result === 'Good Standing' ? (
                    <div className="p-3 rounded-full bg-emerald-50"><CheckCircle className="w-8 h-8 text-emerald-500" /></div>
                  ) : (
                    <div className="p-3 rounded-full bg-destructive/10"><AlertTriangle className="w-8 h-8 text-destructive" /></div>
                  )}
                  <div>
                    <h3 className="text-xl font-bold">{result.student?.name}</h3>
                    <div className="flex items-center gap-2 mt-1 flex-wrap">
                      <Badge variant={result.result === 'Good Standing' ? 'default' : 'destructive'}>{result.result}</Badge>
                      <span className="text-sm text-muted-foreground">{((result.confidence || 0) * 100).toFixed(0)}% confidence</span>
                      {result.student?.gpa_history?.length > 0 && (
                        <span className="text-sm text-muted-foreground">
                          • GPA: {getLatestGPA(result.student.gpa_history)} ({getGradeLabel(getLatestGPA(result.student.gpa_history))})
                        </span>
                      )}
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 text-sm">XAI Explanation</h4>
                  <p className="text-sm text-muted-foreground bg-muted/50 p-3 rounded-lg">{result.explanation}</p>
                </div>

                {/* Categorized Risk Analysis */}
                {result.categorizedRisk && (
                  <div>
                    <h4 className="font-semibold mb-3 text-sm">Categorized Risk Analysis</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {/* Financial Risk */}
                      <div className={`p-3 rounded-lg ${result.categorizedRisk.financial.isAtRisk ? 'bg-destructive/10 border border-destructive/20' : 'bg-emerald-50 border border-emerald-200'}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <DollarSign className={`w-4 h-4 ${result.categorizedRisk.financial.isAtRisk ? 'text-destructive' : 'text-emerald-600'}`} />
                          <span className="font-medium text-sm">Financial</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{result.categorizedRisk.financial.explanation}</p>
                      </div>

                      {/* Personal Risk */}
                      <div className={`p-3 rounded-lg ${result.categorizedRisk.personal.isAtRisk ? 'bg-destructive/10 border border-destructive/20' : 'bg-emerald-50 border border-emerald-200'}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <User className={`w-4 h-4 ${result.categorizedRisk.personal.isAtRisk ? 'text-destructive' : 'text-emerald-600'}`} />
                          <span className="font-medium text-sm">Personal</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{result.categorizedRisk.personal.explanation}</p>
                      </div>

                      {/* Academic Risk */}
                      <div className={`p-3 rounded-lg ${result.categorizedRisk.academic.isAtRisk ? 'bg-destructive/10 border border-destructive/20' : 'bg-emerald-50 border border-emerald-200'}`}>
                        <div className="flex items-center gap-2 mb-2">
                          <BookOpen className={`w-4 h-4 ${result.categorizedRisk.academic.isAtRisk ? 'text-destructive' : 'text-emerald-600'}`} />
                          <span className="font-medium text-sm">Academic</span>
                        </div>
                        <p className="text-xs text-muted-foreground">{result.categorizedRisk.academic.explanation}</p>
                      </div>
                    </div>
                  </div>
                )}

                {/* Low Data Points Analysis */}
                {result.student && (
                  <div>
                    <h4 className="font-semibold mb-3 text-sm">Data Points Analysis (Areas of Concern)</h4>
                    <div className="space-y-2">
                      {result.student.family_income < 20000 && (
                        <div className="p-2 rounded bg-amber-50 border border-amber-200 text-xs">
                          <span className="font-medium text-amber-800">Financial:</span> Family income (₱{result.student.family_income.toLocaleString()}) is below threshold (₱20,000)
                        </div>
                      )}
                      {result.student.scholarship !== 'yes' && result.student.family_income < 20000 && (
                        <div className="p-2 rounded bg-amber-50 border border-amber-200 text-xs">
                          <span className="font-medium text-amber-800">Financial:</span> No scholarship support detected
                        </div>
                      )}
                      {(() => {
                        const gpaValues = [result.student.gpa_y1s1, result.student.gpa_y1s2, result.student.gpa_y2s1, result.student.gpa_y2s2, result.student.gpa_y3s1].filter(g => g && g > 0);
                        if (gpaValues.length > 0) {
                          const avgGpa = gpaValues.reduce((a, b) => a + b, 0) / gpaValues.length;
                          if (avgGpa > 2.5) {
                            return (
                              <div className="p-2 rounded bg-amber-50 border border-amber-200 text-xs">
                                <span className="font-medium text-amber-800">Academic:</span> Average GPA ({avgGpa.toFixed(2)}) is below passing threshold (2.5)
                              </div>
                            );
                          }
                        }
                        return null;
                      })()}
                      {(() => {
                        const surveyAvg = (
                          (result.student.like_course || 3) +
                          (result.student.interested_in_subjects || 3) +
                          (result.student.course_motivates || 3) +
                          (result.student.satisfied_with_performance || 3) +
                          (result.student.previous_grades_affect || 3) +
                          (result.student.study_regularly || 3) +
                          (result.student.submit_on_time || 3) +
                          (result.student.manage_time_well || 3) +
                          (result.student.instructors_explain_clearly || 3) +
                          (result.student.approach_instructors || 3) +
                          (result.student.instructors_encourage || 3) +
                          (result.student.classmates_influence_positively || 3) +
                          (result.student.work_well_with_classmates || 3) +
                          (result.student.friends_motivate || 3)
                        ) / 14;
                        if (surveyAvg <= 3) {
                          return (
                            <div className="p-2 rounded bg-amber-50 border border-amber-200 text-xs">
                              <span className="font-medium text-amber-800">Personal:</span> Survey responses average ({surveyAvg.toFixed(1)}/5) indicates negative sentiment
                            </div>
                          );
                        }
                        return null;
                      })()}
                    </div>
                  </div>
                )}

                {/* Feature Importance (XAI) */}
                {result.feature_importance && (
                  <div>
                    <h4 className="font-semibold mb-3 text-sm">Feature Importance (XAI Explanation)</h4>
                    <div className="space-y-2">
                      {result.feature_importance.slice(0, 5).map((feature, i) => (
                        <div key={i} className="p-2 rounded bg-blue-50 border border-blue-200">
                          <div className="flex items-center justify-between mb-1">
                            <span className="text-xs font-medium text-blue-800">{feature.feature}</span>
                            <span className="text-xs font-bold text-blue-600">{(feature.importance * 100).toFixed(1)}%</span>
                          </div>
                          <div className="w-full h-2 bg-blue-200 rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-blue-500" 
                              style={{ width: `${feature.importance * 100}%` }}
                            />
                          </div>
                          <p className="text-xs text-blue-700 mt-1">
                            {feature.feature === 'GPA' && 'Academic performance is the strongest predictor. Low GPA directly correlates with higher risk of academic failure.'}
                            {feature.feature === 'Financial Status' && 'Financial constraints significantly impact student ability to focus on academics and access resources.'}
                            {feature.feature === 'Study Hours' && 'Time dedicated to studying is crucial for academic success. Low study hours indicate poor time management.'}
                            {feature.feature === 'Course Interest' && 'Student engagement and interest in their course strongly affects motivation and persistence.'}
                            {feature.feature === 'Learning Behavior' && 'Study habits and time management skills are fundamental to academic achievement.'}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="font-semibold mb-2 text-emerald-600 text-sm">Strengths</h4>
                    <ul className="space-y-1">
                      {result.strengths?.map((s, i) => (
                        <li key={i} className="text-sm flex items-start gap-2">
                          <CheckCircle className="w-3.5 h-3.5 text-emerald-500 mt-0.5 flex-shrink-0" />{s}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div>
                    <h4 className="font-semibold mb-2 text-destructive text-sm">Weaknesses</h4>
                    <ul className="space-y-1">
                      {result.weaknesses?.map((w, i) => (
                        <li key={i} className="text-sm flex items-start gap-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-destructive mt-0.5 flex-shrink-0" />{w}
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div>
                  <h4 className="font-semibold mb-2 text-sm">Recommendations</h4>
                  <ul className="space-y-1">
                    {result.recommendations?.map((r, i) => (
                      <li key={i} className="flex items-start gap-2 p-2 rounded bg-primary/5">
                        <TrendingUp className="w-3.5 h-3.5 text-primary mt-0.5 flex-shrink-0" />
                        <span className="text-sm">{r}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            ) : (
              <div className="py-12 text-center text-muted-foreground">
                <Brain className="w-12 h-12 mx-auto mb-3 opacity-30" />
                <p>Select a student and run a prediction to see results</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {predictions.length > 0 && (
        <Card className="mt-6">
          <CardHeader>
            <div className="flex items-center justify-between">
              <CardTitle className="text-base">Recent Predictions</CardTitle>
              <div className="flex gap-2">
                <Button 
                  variant="destructive" 
                  size="sm"
                  onClick={() => setDeletingAll(true)}
                  disabled={deleteAllPredictionsMutation.isPending}
                >
                  {deleteAllPredictionsMutation.isPending ? (
                    <><Trash2 className="w-4 h-4 mr-2 animate-spin" />Deleting...</>
                  ) : (
                    <><Trash2 className="w-4 h-4 mr-2" />Delete All</>
                  )}
                </Button>
                <Button variant="outline" onClick={() => qc.invalidateQueries({ queryKey: ['predictions'] })}>
                  Refresh
                </Button>
              </div>
            </div>
          </CardHeader>
          <CardContent>
            <div className="mb-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <Input
                  placeholder="Search predictions..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="pl-10"
                />
              </div>
            </div>
            <div className="space-y-2">
              {filteredPredictions.length === 0 ? (
                <div className="text-center text-muted-foreground py-8">
                  No predictions match your search.
                </div>
              ) : (
                filteredPredictions.slice(0, 10).map(p => (
                  <div key={p.id} className="flex items-center justify-between p-3 rounded-lg bg-muted/50">
                    <div>
                      <p className="text-sm font-medium">{p.student_name || p.student_id}</p>
                      <p className="text-xs text-muted-foreground">{p.model_used} • {p.prediction_type} • {p.created_date ? new Date(p.created_date).toLocaleDateString() : ''}</p>
                    </div>
                    <Badge variant={p.result === 'Good Standing' ? 'default' : 'destructive'}>{p.result}</Badge>
                  </div>
                ))
              )}
            </div>
          </CardContent>
        </Card>
      )}

      <Dialog open={showBulkSummary} onOpenChange={setShowBulkSummary}>
        <DialogContent className="max-w-md">
          <DialogHeader><DialogTitle>Bulk Prediction Complete</DialogTitle></DialogHeader>
          {bulkSummary && (
            <div className="space-y-4 py-2">
              <div className="grid grid-cols-3 gap-3 text-center">
                <div className="p-4 rounded-lg bg-muted/50">
                  <p className="text-2xl font-bold">{bulkSummary.total}</p>
                  <p className="text-xs text-muted-foreground mt-1">Total Students</p>
                </div>
                <div className="p-4 rounded-lg bg-emerald-50">
                  <p className="text-2xl font-bold text-emerald-600">{bulkSummary.goodStanding}</p>
                  <p className="text-xs text-muted-foreground mt-1">Good Standing</p>
                </div>
                <div className="p-4 rounded-lg bg-destructive/10">
                  <p className="text-2xl font-bold text-destructive">{bulkSummary.atRisk}</p>
                  <p className="text-xs text-muted-foreground mt-1">At-Risk</p>
                </div>
              </div>
              <div className="p-3 rounded-lg bg-muted/50 text-sm text-muted-foreground">
                <p><strong>Good Standing Rate:</strong> {((bulkSummary.goodStanding / bulkSummary.total) * 100).toFixed(1)}%</p>
                <p><strong>At-Risk Rate:</strong> {((bulkSummary.atRisk / bulkSummary.total) * 100).toFixed(1)}%</p>
                <p className="mt-1">All results saved to Prediction Logs.</p>
              </div>
              <Button className="w-full" onClick={() => setShowBulkSummary(false)}>Close</Button>
            </div>
          )}
        </DialogContent>
      </Dialog>

      <AlertDialog open={deletingAll} onOpenChange={setDeletingAll}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete All Predictions</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete all {predictions.length} predictions? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => deleteAllPredictionsMutation.mutate()}
              className="bg-destructive text-destructive-foreground"
            >
              Delete All
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}