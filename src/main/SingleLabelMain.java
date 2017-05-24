package main;

import classify.ColletClassifier;
import classify.DefaultOrNotClassifier;
import classify.HouppierClassifier;
import classify.RacineClassifier;
import classify.TroncClassifier;
import weka.classifiers.trees.RandomForest;

public class SingleLabelMain {
	
	public static void testRandomForest(String trainFile, String testFile, String outFile) throws Exception{
//		StringBuilder param = new StringBuilder();
		StringBuilder sb = new StringBuilder();
		
		String firstColumn = "Parameter\nPrecision\nRecall\nF-Measure\n";
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		
		for (int i = 5; i < 20; i++){
			rd.setMaxDepth(i);
//			sb.append(i + "\n");
			DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainFile, testFile, rd);
			defaultOrNot.evaluateTest();
			sb.append(defaultOrNot.evaluteTestSet());
			sb.append("\t");
		}
		
//		System.out.println(sb);
		
	}
	public static void main(String[] args) throws Exception {
		
		String trainFile = "E:\\dieu\\fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "E:\\dieu\\fixed\\test_set1_semicolon_Fixed.arff";
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainFile, testFile, rd);
		String [][] result = defaultOrNot.evaluateTest();
		
		for (int j = 0; j < result[0].length; j++)
			System.out.println(result[0][j] + " " + result[1][j]);
		
//		testRandomForest(trainFile, testFile, "");

	}
}
