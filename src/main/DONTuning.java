package main;

import classify.DefaultOrNotClassifier;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SGD;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.RandomForest;

public class DONTuning implements Tuning{
	
	public static void testRandomForest(String trainfile, String testfile) throws Exception{
		StringBuilder sb = new StringBuilder(header);
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		
		for (int i = 5; i < 20; i++){
			rd.setMaxDepth(i);
			sb.append(i + "\t");
			DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, rd);
			sb.append(defaultOrNot.evaluateCrossValidation());
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
			DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, sgo);
			System.out.println(defaultOrNot.evaluateCrossValidation());
			epsilon = epsilon/2;
		}
		*/
		//now try with different values of learning rate L
		double lr = sgo.getLearningRate();
		for (int i = 0; i < 10; i++){
			System.out.println(lr);
			sgo.setLearningRate(lr);
			DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, sgo);
			System.out.println(defaultOrNot.evaluateCrossValidation());
			lr = lr/2;
		}
	}
	
	public static void testBayesNet(String trainfile, String testfile) throws Exception{
		BayesNet bn = new BayesNet();
		bn.setUseADTree(true);
		DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, bn);
		System.out.println(defaultOrNot.evaluateCrossValidation());
	}
	
	public static void testOneR(String trainfile, String testfile) throws Exception{
		OneR or = new OneR();
		DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, or);
		System.out.println(defaultOrNot.evaluateCrossValidation());
	}
	
	public static void testAdaBoostM1(String trainfile, String testfile) throws Exception{
		AdaBoostM1 ada = new AdaBoostM1();
		ada.setClassifier(new BayesNet());
		DefaultOrNotClassifier defaultOrNot = new DefaultOrNotClassifier(trainfile, testfile, ada);
		System.out.println(defaultOrNot.evaluateCrossValidation());
	}
	
	public static void main(String[] args) throws Exception {
		System.out.println("RandomForest:");
		testRandomForest(trainfile, testfile);
		
		System.out.println("SGD: ");
		testSGD(trainfile, testfile);
		
		System.out.println("BayesNet: ");
		testBayesNet(trainfile, testfile);
		
		System.out.println("OneR: ");
		testOneR(trainfile, testfile);
		
		System.out.println("AdaBoostM1: ");
		testAdaBoostM1(trainfile, testfile);
	}
}
