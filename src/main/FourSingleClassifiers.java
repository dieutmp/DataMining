package main;

import classify.ColletClassifier;
import classify.HouppierClassifier;
import classify.RacineClassifier;
import classify.TroncClassifier;
import weka.classifiers.functions.SGD;
import weka.classifiers.rules.OneR;
import weka.classifiers.trees.RandomForest;

public class FourSingleClassifiers {
	public static void main(String[] args) throws Exception {
		String trainFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_set1_semicolon_Fixed.arff";
//		SGD rd = new SGD();
		RandomForest rd = new RandomForest();
		rd.setNumIterations(10);
		rd.setMaxDepth(7);
		
		ColletClassifier collet = new ColletClassifier(trainFile, testFile, rd);
		HouppierClassifier houppier = new HouppierClassifier(trainFile, testFile, new SGD());
		TroncClassifier tronc = new TroncClassifier(trainFile, testFile, rd);
		RacineClassifier racine = new RacineClassifier(trainFile, testFile, new OneR());
		
		String[][] colletResult = collet.evaluateTest();
		String[][] houppierResult = houppier.evaluateTest();
		String[][] troncResult = tronc.evaluateTest();
		String[][] racineResult = racine.evaluateTest();
		
		int correct = 0;
		int numberTestInstance = colletResult[0].length;
		for (int i = 0; i < numberTestInstance; i++){
			String actual = colletResult[0][i] + houppierResult[0][i] +
							troncResult[0][i]+ racineResult[0][i];
			
			String predicted = colletResult[1][i] + houppierResult[1][i] +
					troncResult[1][i]+ racineResult[1][i];
			
			if (actual.equals(predicted))
				correct++;
			else
			System.out.println(actual + " " + predicted);
			
		}
		System.out.println("Correct " + correct);
		System.out.println(correct/numberTestInstance);
	}
}
