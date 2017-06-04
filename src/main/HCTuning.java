package main;

import classify.HouppierClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SGD;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.RandomForest;

public class HCTuning implements Tuning{
	public static void testRandomForest(String trainfile, String testfile) throws Exception{
		StringBuilder sb = new StringBuilder(header);
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		
		for (int i = 5; i < 20; i++){
			rd.setMaxDepth(i);
			sb.append(i + "\t");
			HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, rd);
			sb.append(houppier.evaluateCrossValidation());
			sb.append("\n");
		}
		
		System.out.println(sb);
		
	}
	
	public static void testSGD(String trainfile, String testfile) throws Exception{
		SGD sgo = new SGD();
		/*
		 * try these values of epsilon, the result are the same 
		double epsilon = 0.001;
		while (epsilon > 0.0001){
			System.out.println(epsilon);
			sgo.setEpsilon(epsilon);
			HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, sgo);
			System.out.println(houppier.evaluteTestSet());
			epsilon = epsilon/2;
		}
		*/
		//now try with different values of learning rate L
		double lr = sgo.getLearningRate();
		for (int i = 0; i < 10; i++){
			System.out.println(lr);
			sgo.setLearningRate(lr);
			HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, sgo);
			System.out.println(houppier.evaluateCrossValidation());
			lr = lr/2;
		}
	}
	
	public static void testBayesNet(String trainfile, String testfile) throws Exception{
		BayesNet bn = new BayesNet();
		bn.setUseADTree(true);
		HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, bn);
		System.out.println(houppier.evaluateCrossValidation());
	}
	
	public static void testOneR(String trainfile, String testfile) throws Exception{
		OneR or = new OneR();
		HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, or);
		System.out.println(houppier.evaluateCrossValidation());
	}
	
	public static void testAdaBoostM1(String trainfile, String testfile) throws Exception{
		AdaBoostM1 ada = new AdaBoostM1();
		ada.setClassifier(new BayesNet());
		HouppierClassifier houppier = new HouppierClassifier(trainfile, testfile, ada);
		System.out.println(houppier.evaluateCrossValidation());
	}
	
	public static void main(String[] args) throws Exception {
//		System.out.println("Random forest:");
//		testRandomForest(trainfile, testfile);
//		System.out.println("SGD");
//		testSGD(trainfile, testfile);
		System.out.println("BayesNet: ");
		testBayesNet(trainfile, testfile);
		System.out.println("OneR");
		testOneR(trainfile, testfile);
		System.out.println("AdaBoostM1");
		testAdaBoostM1(trainfile, testfile);
	}
}
