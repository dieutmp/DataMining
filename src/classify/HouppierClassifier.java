package classify;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.FilteredClassifier;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;

public class HouppierClassifier{
	Instances train;
	Instances test;
	Evaluation eval;
	FilteredClassifier fc;
	
	public HouppierClassifier(String trainFile, String testFile, Classifier cls) throws Exception{
		//loading to Instances for train
		DataSource sourceTrain = new DataSource(trainFile);
		train = sourceTrain.getDataSet();
		
		//loading to Instances for test
		DataSource sourceTest = new DataSource(testFile);
		test = sourceTest.getDataSet();
		
		
		//find class index and indices to be removed
		int classIndex = 0;
		int[] removedIndices = new int[4]; //always remove 4 indices that are classes
		int j = 0;
		for (int i = 0; i < train.numAttributes(); i++){
			String name = train.attribute(i).name().toLowerCase();
			if (name.startsWith("houppier")){
				classIndex = i;
			}else if (name.startsWith("default.or.not") || name.startsWith("collet") || name.startsWith("racine") || name.startsWith("tronc")){
				removedIndices[j] = i;
				j++;
			} 
			
		}		
		train.setClassIndex(classIndex);
		test.setClassIndex(classIndex);
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(removedIndices);
		
		fc = new FilteredClassifier();
		fc.setFilter(rm);
		fc.setClassifier(cls);
		
//		String options = String.format("-t %s -T %s -c 18 -x 4 -s 1 -i", 
//										trainFile, testFile, classIndex);
//		String[] o = Utils.splitOptions(options);
//		for (String s : o){
//			System.out.println(s);
//		}
//		Evaluation.evaluateModel(fc, o);
	
	}
	
	
	public String evaluateCrossValidation() throws Exception{
		StringBuilder sb = new StringBuilder();
		eval = new Evaluation(train);
		Random rand = new Random(1);  // using seed = 1
		int folds = 10;
		eval = new Evaluation(train);
		eval.crossValidateModel(fc, train, folds, rand);
		sb.append(eval.toSummaryString("CrossValidation Result", true));
		sb.append(eval.toClassDetailsString());
		sb.append(eval.toMatrixString());
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