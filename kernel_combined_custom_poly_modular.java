import org.shogun.*;
import org.jblas.*;

import static org.shogun.LabelsFactory.to_binary;

public class kernel_combined_custom_poly_modular {
	static {
		//System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) {
		//modshogun.init_shogun_with_defaults();
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		
		DoubleMatrix traindata_real = Load.load_numbers("../data/iris_train_data");
		DoubleMatrix testdata_real = Load.load_numbers("../data/iris_test_data");
		//DoubleMatrix train_data_string = traindata_real.getRows(new int[]{1,2});
		DoubleMatrix trainlab = Load.load_labels("../data/iris_train_labels");

		CombinedKernel kernel = new CombinedKernel();
		CombinedFeatures feats_train = new CombinedFeatures();
		
		RealFeatures tfeats = new RealFeatures(traindata_real);
		LinearKernel custom_linear = new LinearKernel();
		custom_linear.init(feats_train, feats_train);
		DoubleMatrix K = custom_linear.get_kernel_matrix();
		kernel.append_kernel(new CustomKernel(K));
		

		RealFeatures subkfeats_train = new RealFeatures(traindata_real);
		feats_train.append_feature_obj(subkfeats_train);
		GaussianKernel subkernel = new GaussianKernel();
		kernel.append_kernel(subkernel);
		kernel.init(feats_train, feats_train);

		BinaryLabels labels = new BinaryLabels(trainlab);

		LibSVM svm = new LibSVM(C, kernel, labels);
		svm.train();

		CombinedKernel kernel_pred = new CombinedKernel();
		CombinedFeatures feats_pred = new CombinedFeatures();

		RealFeatures pfeats = new RealFeatures(testdata_real);
		GaussianKernel tkernel_pred = new GaussianKernel(10,3);
		tkernel_pred.init(tfeats, pfeats);
		//LinearKernel custom_linear = new LinearKernel();
		//custom_linear.init(feats_train, feats_train);
		
		//DoubleMatrix KK = tkernel.get_kernel_matrix();
		//kernel_pred.append_kernel(new CustomKernel(KK));

		RealFeatures subkfeats_test = new RealFeatures(testdata_real);
		feats_pred.append_feature_obj(subkfeats_train);
		PolyKernel subkernel_pred = new PolyKernel(10,2);
		kernel_pred.append_kernel(subkernel_pred);

		kernel_pred.init(feats_train, feats_pred);

		svm.set_kernel(kernel_pred);
		to_binary(svm.apply());
		DoubleMatrix km_train=kernel.get_kernel_matrix();
		System.out.println(km_train.toString());

	}
}
