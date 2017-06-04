package multilabel;

import util.Utilizer;
import weka.classifiers.bayes.BayesNet;
import weka.classifiers.functions.SGD;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;

public class MultiLabelCTuning {
	public static void testRandomForest(Instances train, Instances test) throws Exception{
		StringBuilder sb = new StringBuilder();
		String firstColumn = "Parameter\nPrecision\nRecall\nF-Measure\n";
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		
		for (int i = 5; i < 20; i++){
			rd.setMaxDepth(i);
			sb.append(i + "\n");
			MultilabelClassifier chrt = new MultilabelClassifier(train, test, rd);
			sb.append(chrt.evaluateCrossValidation());
			sb.append("\t");
		}
		
		System.out.println(sb);
		
	}
	
	public static void testSGD(Instances train, Instances test) throws Exception{
		SGD sgo = new SGD();
		/*
		 * try these values of epsilon, the result are the same 
		double epsilon = 0.001;
		while (epsilon > 0.0001){
			System.out.println(epsilon);
			sgo.setEpsilon(epsilon);
			MultilabelClassifier chrt = new MultilabelClassifier(train, test, sgo);
			System.out.println(chrt.evaluteTestSet());
			epsilon = epsilon/2;
		}
		*/
		//now try with different values of learning rate L
		double lr = sgo.getLearningRate();
		for (int i = 0; i < 10; i++){
			System.out.println(lr);
			sgo.setLearningRate(lr);
			MultilabelClassifier chrt = new MultilabelClassifier(train, test, sgo);
			System.out.println(chrt.evaluateCrossValidation());
			lr = lr/2;
		}
	}
	
	public static void testBayesNet(Instances train, Instances test) throws Exception{
		BayesNet bn = new BayesNet();
		bn.setUseADTree(true);
		MultilabelClassifier chrt = new MultilabelClassifier(train, test, bn);
		System.out.println(chrt.evaluateCrossValidation());
	}
	
	public static void testOneR(Instances train, Instances test) throws Exception{
		OneR or = new OneR();
		MultilabelClassifier chrt = new MultilabelClassifier(train, test, or);
		System.out.println(chrt.evaluateCrossValidation());
	}
	
	public static void testAdaBoostM1(Instances train, Instances test) throws Exception{
		AdaBoostM1 ada = new AdaBoostM1();
//		ada.setClassifier(new BayesNet());
//		ada.setClassifier(new OneR());
		MultilabelClassifier chrt = new MultilabelClassifier(train, test, ada);
		System.out.println(chrt.evaluateCrossValidation());
	}
	
	public static void main(String[] args) throws Exception {
		String trainFile ="C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_mergedClass.arff";
		String testFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_set1_semicolon_Fixed.arff";
		Instances train = Utilizer.loadInstancesFromArff(trainFile);
		Instances test  = Utilizer.loadInstancesFromArff(testFile);
//		System.out.println("Random forest:");
//		testRandomForest(train, test);
//		System.out.println("SGD");
//		testSGD(train, test);
		System.out.println("BayesNet: ");
		testBayesNet(train, test);
		System.out.println("OneR");
		testOneR(train, test);
		System.out.println("AdaBoostM1");
		testAdaBoostM1(train, test);
	}
}
