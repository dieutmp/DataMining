package multilabel;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class MultiLabelClassifier {
	public static void main(String[] args) throws Exception {
		String trainFile = "";
		String testFile = "";
		
		DataSource sourceTrain = new DataSource(trainFile);
		Instances train = sourceTrain.getDataSet();
		
		//loading to Instances for test
		DataSource sourceTest = new DataSource(testFile);
		Instances test = sourceTest.getDataSet();
	}
}
