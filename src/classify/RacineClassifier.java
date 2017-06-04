package classify;

import util.Utilizer;
import weka.classifiers.Classifier;
import weka.classifiers.meta.FilteredClassifier;
import weka.filters.unsupervised.attribute.Remove;

public class RacineClassifier extends AbstractClassifier{
	
	public RacineClassifier(String trainfile, String testfile, Classifier cls) throws Exception{
		this.train = Utilizer.loadInstancesFromArff(trainfile);
		this.test = Utilizer.loadInstancesFromArff(testfile);
		
		//find class index and indices to be removed
		int classIndex = 0;
		int[] removedIndices = new int[4]; //always remove 4 indices that are classes
		int j = 0;
		for (int i = 0; i < train.numAttributes(); i++){
			String name = train.attribute(i).name().toLowerCase();
			if (name.startsWith("racine")){
				classIndex = i;
			}else if (name.startsWith("default.or.not") || name.startsWith("collet") || name.startsWith("houppier") || name.startsWith("tronc")){
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
	}
	
}