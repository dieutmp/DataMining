package classify;

import java.io.File;

import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.FilteredClassifier;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.NominalToString;
import weka.filters.unsupervised.attribute.NumericToNominal;
import weka.filters.unsupervised.attribute.Remove;

public class WekaTest{
	public static void main(String[] args) throws Exception {
		String trainFile = "E:\\dieu\\new_arff_files\\train_set1_semicolon - New Pandu.arff";
		DataSource sourceTrain = new DataSource(trainFile);
		
		Instances train = sourceTrain.getDataSet();
		train.setClassIndex(18);
//		System.out.println("blank line: ");
//		System.out.println(train.attribute(5));
//		
//		System.out.println(train.numAttributes());
//		for (int i = 0; i < train.numAttributes(); i++){
//			System.out.println(train.attribute(i));
//		}
		
		String testFile = "E:\\dieu\\new_arff_files\\test_set1_semicolon - New.arff";
		DataSource sourceTest = new DataSource(testFile);
		Instances test = sourceTest.getDataSet();
		test.setClassIndex(18);
		
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(new int[]{20, 21, 22, 19});
		
		FilteredClassifier fc = new FilteredClassifier();
		RandomForest randomForest = new RandomForest();
		randomForest.setNumIterations(10);
		
		
		fc.setFilter(rm);
		fc.setClassifier(randomForest);
		
		for (String s : randomForest.getOptions())
			System.out.println(s);
		 // train and make predictions
		 fc.buildClassifier(train);
		 
		 
		 int correct = 0;
		 
		 for (int i = 0; i < test.numInstances(); i++) {
		   double pred = fc.classifyInstance(test.instance(i));
//		   System.out.print("ID: " + test.instance(i).value(0));
//		   System.out.print(", actual: " + test.classAttribute().value((int) test.instance(i).classValue()));
//		   System.out.println(", predicted: " + test.classAttribute().value((int) pred));
		   if (test.classAttribute().value((int) test.instance(i).classValue()) == test.classAttribute().value((int) pred))
			   correct += 1;
		 }
		
		System.out.println("correct: " + correct + " total: " + test.numInstances());
	}
}