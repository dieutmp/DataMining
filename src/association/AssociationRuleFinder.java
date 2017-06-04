package association;

import weka.associations.Apriori;
import weka.core.DenseInstance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

public class AssociationRuleFinder {
	public static void diameterVsAge(Instances data){
		Remove rm = new Remove();
		for (int i = 0; i < data.numAttributes(); i++){
			
		}
	
	}
	public static void main(String[] args) throws Exception {
		String trainFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\train_set1_semicolon_Fixed.arff";
		String testFile = "C:\\Users\\21609450t\\IT4BI\\DM\\phase2\\Fixed\\test_set1_semicolon_Fixed.arff";
		
		DataSource sourceTrain = new DataSource(trainFile);
		Instances train = sourceTrain.getDataSet();
		
		int[] removedIndices = new int[7]; //remove 2 indices of coordinates
		int j = 0;
		for (int i = 0; i < train.numAttributes(); i++){
			String name = train.attribute(i).name().toLowerCase();
			if (name.startsWith("coord")
//					|| name.startsWith("default.or.not") || name.startsWith("houppier")
//								|| name.startsWith("racine") || name.startsWith("tronc") 
								|| name.startsWith("stadededeveloppement")){
				removedIndices[j] = i;
				j++;
			} 
			
		}		
		Remove rm = new Remove();
		rm.setAttributeIndicesArray(removedIndices);
		rm.setInputFormat(train);
		Instances filteredTrain = Filter.useFilter(train, rm);
		
		Apriori model = new Apriori();
		model.buildAssociations(filteredTrain);
		System.out.println(model);
	}
}
