package util;

import java.io.File;
import java.io.IOException;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Debug.Random;
import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Add;
import weka.filters.unsupervised.attribute.Remove;

public class Utilizer {
	
	public static void saveInstancesToArffFile(Instances data, String outFile) throws IOException{
		ArffSaver saver = new ArffSaver();
		saver.setInstances(data);
		saver.setFile(new File(outFile));
		saver.writeBatch();
	}
	
	public static Instances loadInstancesFromCsv(String csvFile) throws IOException{
		CSVLoader loader = new CSVLoader();
		loader.setSource(new File(csvFile));
		return loader.getDataSet();
		
	}
	
	public static Instances loadInstancesFromArff(String arffFile) throws Exception{
		DataSource sourceTrain = new DataSource(arffFile);
		return sourceTrain.getDataSet();
	}
	
	public static void mergeInstances() throws Exception{
		String trainFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_set1_semicolon_Fixed.arff";
		
		Instances train = loadInstancesFromArff(trainFile);
		Instances test = loadInstancesFromArff(testFile);
		
		for (int i = 0; i < test.numInstances(); i++)
			train.add(test.get(i));
		Instances data = new Instances(train);
		saveInstancesToArffFile(data, "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\all_data.arff");
	}
	
	/*
	 * Split data to train and test
	 * Reference: http://weka.wikispaces.com/Generating+cross-validation+folds+(Java+approach)
	 * @param data: instances to be splitted to train and test
	 * @param seed:  seed for randomizing the data
	 * @param folds: number of folds to generate, >=2
	 */
	public static Instances[] splitInstances(Instances data, int seed, int folds){
		Instances[] train_test = new Instances[2];
		Random rand = new Random(seed);   // create seeded number generator
		Instances randData = new Instances(data);   // create copy of original data
		randData.randomize(rand); // randomize data with number generator
		
		randData.stratify(folds); //stratify to 
		train_test[0] = randData.trainCV(folds, 0);
		train_test[1] = randData.testCV(folds, 0);
		
		return train_test;
		
	}
	
	/*
	 * transform original data to the new one that has class attribute as 
	 * a merge of collet-houppier-racine-tronc (chrt)
	 */
	public static Instances transform(Instances data) throws Exception{
		int n = data.numAttributes();
		int[] removedIndices = new int[5]; 
		int j = 0;
		for (int i = 0; i < data.numAttributes(); i++){
			String name = data.attribute(i).name().toLowerCase();
			if (name.startsWith("collet") || name.startsWith("houppier") || name.startsWith("racine") || name.startsWith("tronc")){
				removedIndices[j] = i;
				j++;
			} else if (name.startsWith("default.or.not")){
				removedIndices[4] = i;
			}
		}		
		
		Add add = new Add();
		add.setAttributeIndex("last");
		add.setAttributeName("chrt");
		int a, b, c, d;
		StringBuilder sb = new StringBuilder("");
		for (a=0;a<2;a++)
			for (b=0;b<2;b++)
				for (c=0;c<2;c++)
					for (d=0;d<2;d++)
						sb.append(""+a+b+c+d+",");
		add.setNominalLabels(sb.toString().substring(0, sb.toString().length()-1));
		System.out.println(sb.toString().substring(0, sb.toString().length()-1));
		add.setInputFormat(data);
		Instances newTrain = Filter.useFilter(data, add);
		
		for (int i=0; i<newTrain.numInstances(); i++){
			sb = new StringBuilder("");
			for (j=0; j<4; j++)
				sb.append((int)newTrain.instance(i).value(removedIndices[j]));
			newTrain.instance(i).setValue(n, sb.toString());
					
		}
		
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(removedIndices);
		rm.setInputFormat(newTrain);
		Instances result = Filter.useFilter(newTrain, rm);
		return result;
	}
	
	
	public static void main(String[] args) throws Exception {
		
		String trainFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_set1_semicolon_Fixed.arff";
		Instances trainTemp = loadInstancesFromArff(trainFile);
		Instances testTemp = loadInstancesFromArff(testFile);
		
		Instances train = transform(trainTemp);
		Instances test = transform(testTemp);
	
		saveInstancesToArffFile(train, "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_mergedClass.arff");
		saveInstancesToArffFile(test, "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_mergedClass.arff");
		
	}
}
