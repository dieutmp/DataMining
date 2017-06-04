package classify;

import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Debug.Random;
import weka.core.Instances;

public abstract class AbstractClassifier {
	Instances train;
	Instances test;
	Evaluation eval;
	FilteredClassifier fc;
	
	private static double round(double d){
		return Math.round(d*100.0)/100.0;
	}
	
	public String evaluateCrossValidation() throws Exception{
		StringBuilder sb = new StringBuilder();
		eval = new Evaluation(train);
		Random rand = new Random(1);  // using seed = 1
		int folds = 10;
		eval = new Evaluation(train);
		eval.crossValidateModel(fc, train, folds, rand);
		sb.append(round(eval.pctCorrect()) + "\t");
		sb.append(round(eval.precision(1)) + "\t");
		sb.append(round(eval.precision(0)) + "\t");
		sb.append(round(eval.recall(1)) + "\t");
		sb.append(round(eval.recall(0)) + "\t");
		sb.append(round(eval.fMeasure(1))+ "\t");
		sb.append(round(eval.fMeasure(0)) + "\n");
		return sb.toString();
	}
	
	public String evaluteTestSet() throws Exception{
		StringBuilder sb = new StringBuilder();
		fc.buildClassifier(train);
		eval = new Evaluation(train);
		eval.evaluateModel(fc, test);
		sb.append(eval.toSummaryString("Test Result", true));
		sb.append(eval.toClassDetailsString());
		sb.append(eval.toMatrixString());
		return sb.toString();
	}
	
	public String[][] evaluateTest() throws Exception{
		fc.buildClassifier(train);
		int numInstances = test.numInstances();
		int count = 0;
		String[][] result = new String[2][numInstances];
		for (int i = 0; i < numInstances; i++) {
			   double pred = fc.classifyInstance(test.instance(i));
			   String actual = test.classAttribute().value((int) test.instance(i).classValue());
			   String predicted = test.classAttribute().value((int) pred);
			   if (actual == predicted)
				   count ++;
			   result[0][i] = actual;
			   result[1][i] = predicted;
			   
			 }
		System.out.println("correct " + count);
		System.out.println("in total: " + test.numInstances());
		return result;
	}
	
}
