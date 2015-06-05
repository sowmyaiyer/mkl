import static org.shogun.LabelsFactory.to_binary;

import org.shogun.*;
import org.jblas.*;

public class EnhancerFinder {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 2;

		DoubleMatrix traindata_real_all = Load.load_numbers("../data/magic_train_features.1000.txt");
		DoubleMatrix testdata_real_all = Load.load_numbers("../data/magic_test_features.1000.txt");
		DoubleMatrix trainlab = Load.load_labels("../data/magic_train_labels.1000.txt");
		
		System.out.println(traindata_real_all.getRows());
		
		RealFeatures train_features = new RealFeatures(traindata_real_all);	
		RealFeatures test_features = new RealFeatures(testdata_real_all);
	
		LinearKernel gaussianKernel = new LinearKernel();
		gaussianKernel.init(train_features, train_features);
	    DoubleMatrix K_train = gaussianKernel.get_kernel_matrix();
	    
	    System.out.println("gaussian kernel done");

		BinaryLabels labels = new BinaryLabels(trainlab);
		LibSVM svm = new LibSVM(C, gaussianKernel, labels);
		svm.train();
		BinaryLabels test_labels = to_binary(svm.apply(test_features));
		
        System.out.println(test_labels.get_labels().toString());
        System.out.println(test_labels.get_values().toString());
        test_labels.scores_to_probabilities();
        System.out.println(test_labels.get_values().toString());

	}
}
