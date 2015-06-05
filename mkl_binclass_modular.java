import static org.shogun.LabelsFactory.to_binary;

import org.shogun.*;
import org.jblas.*;

public class mkl_binclass_modular {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 2;

		DoubleMatrix traindata_real_all = Load.load_numbers("data/magic_train.1000.txt");
		DoubleMatrix testdata_real_all = Load.load_numbers("data/magic_test.1000.txt");
		DoubleMatrix trainlab = Load.load_labels("data/magic_train_labels.1000.txt");
		
		System.out.println(traindata_real_all.getRows());
		
		DoubleMatrix sub_train_data_string = traindata_real_all.getRows(new int[]{0,1,2,3,4});
		DoubleMatrix sub_train_data_numeric = traindata_real_all.getRows(new int[]{5,6,7,8});
		RealFeatures train_features_string = new RealFeatures(sub_train_data_string);
		RealFeatures train_features_numeric = new RealFeatures(sub_train_data_numeric);
		CombinedFeatures combined_features_train = new CombinedFeatures();
		combined_features_train.append_feature_obj(train_features_string);
		combined_features_train.append_feature_obj(train_features_numeric);
		System.out.println("train features done");

		DoubleMatrix sub_test_data_string = testdata_real_all.getRows(new int[]{0,1,2,3,4});
		DoubleMatrix sub_test_data_numeric = testdata_real_all.getRows(new int[]{5,6,7,8});
		RealFeatures test_features_string = new RealFeatures(sub_test_data_string);
		RealFeatures test_features_numeric = new RealFeatures(sub_test_data_numeric);
		CombinedFeatures combined_features_test = new CombinedFeatures();
		combined_features_test.append_feature_obj(test_features_string);
		combined_features_test.append_feature_obj(test_features_numeric);
		System.out.println("test features done");
				

		GaussianKernel gaussianKernel = new GaussianKernel();
		gaussianKernel.init(train_features_string, train_features_string);
	    DoubleMatrix K_train_string_gaussian = gaussianKernel.get_kernel_matrix();
	    gaussianKernel.init(train_features_string, test_features_string);
	    DoubleMatrix K_test_string_gaussian = gaussianKernel.get_kernel_matrix();
	    System.out.println("gaussian kernel done");
	    
	    LinearKernel linearKernel = new LinearKernel();
	    linearKernel.init(train_features_numeric, train_features_numeric);
	    DoubleMatrix K_train_numeric_linear = linearKernel.get_kernel_matrix();
	    linearKernel.init(train_features_numeric, test_features_numeric);
	    DoubleMatrix K_test_numeric_linear = linearKernel.get_kernel_matrix();
	    System.out.println("linear kernel done");
	    

		CombinedKernel kernel = new CombinedKernel();
		kernel.append_kernel(gaussianKernel);
		kernel.append_kernel(linearKernel);

		kernel.init(combined_features_train, combined_features_train);
		System.out.println("combinedKernel inited");

		BinaryLabels labels = new BinaryLabels(trainlab);

		MKLClassification mkl = new MKLClassification();
		mkl.set_mkl_norm(1);
		mkl.set_kernel(kernel);
	    mkl.set_labels(labels);

		mkl.train();
		System.out.println("mkl trained");
	    		
		//CombinedKernel kernel2 = new CombinedKernel();
		//kernel2.append_kernel(gaussianKernel);
		//kernel2.append_kernel(linearKernel);
		
		//kernel2.init(combined_features_train, combined_features_test);
		//System.out.println("kernel2 inited");
				
		//mkl.set_kernel(kernel2);
		BinaryLabels test_labels = to_binary(mkl.apply(combined_features_test));
		System.out.println("mkl applied to test data");

        System.out.println(test_labels.get_labels().toString());
        System.out.println(test_labels.get_values().toString());
        test_labels.scores_to_probabilities();
        System.out.println(test_labels.get_values().toString());

	}
}
