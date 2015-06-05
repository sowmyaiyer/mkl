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
		int mkl_norm = 2;
		boolean reverse = false;
		boolean use_sign = false;

		String[] fm_train_dna = Load.load_dna("data/H1_ATAC_training_sequences.txt");
		String[] fm_test_dna = Load.load_dna("data/H1_ATAC_test_sequences.txt");
		DoubleMatrix trainlab = Load.load_labels("data/H1_ATAC_training_labels.txt");

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

		CommWordStringKernel kernel = new CommWordStringKernel(feats_train, feats_train, use_sign);

		//DoubleMatrix km_train = kernel.get_kernel_matrix();
		//kernel.init(feats_train, feats_test);
		//DoubleMatrix km_test = kernel.get_kernel_matrix();
		BinaryLabels labels = new BinaryLabels(trainlab);
		SVMLight svm = new SVMLight(C, kernel, labels);
		svm.set_epsilon(epsilon);
		//svm.parallel.set_num_threads(num_threads);
		svm.train();

		
		BinaryLabels test_labels = to_binary(svm.apply(feats_test));
		System.out.println("svm applied to test data");

        //System.out.println(test_labels.get_labels().toString());
		FileWriter fw_scores = new FileWriter("scores.txt");
		fw_scores.write(test_labels.get_values().toString());
		fw_scores.close();
		
        test_labels.scores_to_probabilities();
    	FileWriter fw_probs = new FileWriter("probs.txt");
    	fw_probs.write(test_labels.get_values().toString());
    	fw_probs.close();

	}
}
