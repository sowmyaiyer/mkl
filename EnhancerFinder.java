import org.shogun.*;
import org.jblas.*;
import static org.shogun.EAlphabet.DNA;
import static org.shogun.LabelsFactory.to_binary;
import java.io.FileWriter;


public class EnhancerFinder {
	static {
		System.loadLibrary("modshogun");
	}

	public static void main(String argv[]) throws java.io.IOException {
		modshogun.init_shogun_with_defaults();
		int order = 6;
		int gap = 0;
		double width = 2.1;
		double epsilon = 1e-5;
		double C = 1.0;
		int mkl_norm = 1;
		boolean reverse = true;
		boolean use_sign = false;

		String[] fm_train_dna = Load.load_dna("data/H1_ATAC_training_sequences.txt");
		String[] fm_test_dna = Load.load_dna("data/H1_ATAC_test_sequences.txt");
		DoubleMatrix trainlab = Load.load_labels("data/H1_ATAC_training_labels.txt");
		System.out.println("string loaded");
		
		StringCharFeatures charfeat = new StringCharFeatures(DNA);
		charfeat.set_features(fm_train_dna);
		StringWordFeatures feats_train = new StringWordFeatures(charfeat.get_alphabet());
		feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
		SortWordString preproc = new SortWordString();
		preproc.init(feats_train);
		feats_train.add_preprocessor(preproc);
		feats_train.apply_preprocessor();

		StringCharFeatures charfeat_test = new StringCharFeatures(DNA);
		charfeat_test.set_features(fm_test_dna);
		StringWordFeatures feats_test = new StringWordFeatures(charfeat.get_alphabet());
		feats_test.obtain_from_char(charfeat_test, order-1, order, gap, reverse);
		feats_test.add_preprocessor(preproc);
		feats_test.apply_preprocessor();

		CommWordStringKernel stringkernel = new CommWordStringKernel(feats_train, feats_train, use_sign);
		stringkernel.set_use_dict_diagonal_optimization(false);
		stringkernel.init(feats_train, feats_train);
		
		
		DoubleMatrix traindata_real = Load.load_numbers("data/H1_ATAC_training_numeric_scores.txt");
		DoubleMatrix testdata_real = Load.load_numbers("data/H1_ATAC_test_numeric_scores.txt");
		
		RealFeatures train_features_numeric = new RealFeatures(traindata_real);
		RealFeatures test_features_numeric = new RealFeatures(testdata_real);
		System.out.println("numbers loaded");

		NormOne normone_preproc = new NormOne();
		normone_preproc.init(train_features_numeric);
		train_features_numeric.add_preprocessor(normone_preproc);
		train_features_numeric.apply_preprocessor();
		test_features_numeric.add_preprocessor(normone_preproc);
		test_features_numeric.apply_preprocessor();
		System.out.println("features set");
		
		LinearKernel linearKernel = new LinearKernel();
		linearKernel.init(train_features_numeric, train_features_numeric);
		
		CombinedFeatures combined_features_train = new CombinedFeatures();
		combined_features_train.append_feature_obj(feats_train);
		combined_features_train.append_feature_obj(train_features_numeric);
		
		CombinedFeatures combined_features_test = new CombinedFeatures();
		combined_features_test.append_feature_obj(feats_test);
		combined_features_test.append_feature_obj(test_features_numeric);
		System.out.println("features combined");
		
		CombinedKernel kernel = new CombinedKernel();
		kernel.append_kernel(stringkernel);
		kernel.append_kernel(linearKernel);
		kernel.init(combined_features_train, combined_features_train);

		System.out.println("kernels combined");
		
		BinaryLabels labels = new BinaryLabels(trainlab);
		MKLClassification mkl = new MKLClassification();
		mkl.set_mkl_norm(1);
		mkl.set_kernel(kernel);
	    mkl.set_labels(labels);

	    mkl.train();
		
		System.out.println("mkl trained");
		
		BinaryLabels test_labels = to_binary(mkl.apply(combined_features_test));
		
		System.out.println("svm applied to test data");

       
		FileWriter fw_scores = new FileWriter("scores.txt");
		
		fw_scores.write(test_labels.get_values().toString().replace("[","").replace("]","").replace(", ","\n").replace(",", ""));
		fw_scores.close();
		
        test_labels.scores_to_probabilities();
    	FileWriter fw_probs = new FileWriter("probs.txt");
    	fw_probs.write(test_labels.get_values().toString());
    	fw_probs.close();

	}
}
