package main;

import classify.ColletClassifier;
import classify.DefaultOrNotClassifier;
import classify.HouppierClassifier;
import classify.RacineClassifier;
import classify.TroncClassifier;
import weka.classifiers.trees.RandomForest;

public class AlgoTests {
	
	public static void testRandomForest(String trainFile, String testFile, String outFile) throws Exception{
//		StringBuilder param = new StringBuilder();
		StringBuilder sb = new StringBuilder();
		
		String firstColumn = "Parameter\nPrecision\nRecall\nF-Measure\n";
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		
		for (int i = 5; i < 20; i++){
			rd.setMaxDepth(i);
			sb.append(i + "\n");
			DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainFile, testFile, rd);
			sb.append(defaultOrNot.evaluteTestSet());
			sb.append("\t");
		}
		
		System.out.println(sb);
		
	}
	public static void main(String[] args) throws Exception {
		
		String trainFile = "E:\\dieu\\fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "E:\\dieu\\fixed\\test_set1_semicolon_Fixed.arff";
		testRandomForest(trainFile, testFile, "");
		//test with RandomForest
		/*
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		rd.setMaxDepth(0);
		
		

		DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainFile, testFile, rd);
		System.out.println(defaultOrNot.evaluateCrossValidation());
//		System.out.println(defaultOrNot.evaluteTestSet());
		
		ColletClassifier collet = new ColletClassifier(trainFile, testFile, rd);
		System.out.println(collet.evaluateCrossValidation());
//		System.out.println(collet.evaluteTestSet());
		
		HouppierClassifier houppier = new HouppierClassifier(trainFile, testFile, rd);
		System.out.println(houppier.evaluateCrossValidation());
//		System.out.println(houppier.evaluteTestSet());
		
		TroncClassifier tronc = new TroncClassifier(trainFile, testFile, rd);
		System.out.println(tronc.evaluateCrossValidation());
//		System.out.println(tronc.evaluteTestSet());
		
		RacineClassifier racine = new RacineClassifier(trainFile, testFile, rd);
		System.out.println(racine.evaluateCrossValidation());
//		System.out.println(tronc.evaluteTestSet());
		//test with Classify X
		*/
	}
}
