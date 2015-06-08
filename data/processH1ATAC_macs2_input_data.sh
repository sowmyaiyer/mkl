tr '[:lower:]' '[:upper:]' < H1_ATAC_macs2_peak_distal_sequence.only.txt | sed 's/N//g' > tmp
awk '{ if (NR > 1) print $2"\t"$3"\t"$4"\t"$6"\t"$NF}' H1_ATAC_macs2_trainingset.ALL_numeric.txt > H1_ATAC_macs2_trainingset.select_num_columns.txt
paste tmp H1_ATAC_macs2_trainingset.select_num_columns.txt > H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.txt
shuf -n 15000 H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.txt > H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.random15000.txt
head -13000 H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.random15000.txt > H1_ATAC_macs2_training.txt
tail -2000 H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.random15000.txt > H1_ATAC_macs2_test.txt


awk '{ print $1}' H1_ATAC_macs2_training.txt > H1_ATAC_macs2_training_sequences.txt
awk '{ print $2"\t"$3"\t"$4"\t"$5}' H1_ATAC_macs2_training.txt > H1_ATAC_macs2_training_numeric_scores.txt
awk '{ print $NF}' H1_ATAC_macs2_training.txt | sed 's/0/\-1/g' > H1_ATAC_macs2_training_labels.txt

awk '{ print $1}' H1_ATAC_macs2_test.txt > H1_ATAC_macs2_test_sequences.txt
awk '{ print $2"\t"$3"\t"$4"\t"$5}' H1_ATAC_macs2_test.txt > H1_ATAC_macs2_test_numeric_scores.txt
awk '{ print $NF}' H1_ATAC_macs2_test.txt | sed 's/0/\-1/g' > H1_ATAC_macs2_test_labels.txt


#shuf H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.txt > H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.shuffled.txt
#head -115000 H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.shuffled.txt > H1_ATAC_macs2_training.txt
#tail -2923 H1_ATAC_macs2_trainingset_sequence_and_select_num_columns.shuffled.txt > H1_ATAC_macs2_test.txt
#
#awk '{ print $1}' H1_ATAC_macs2_training.txt > H1_ATAC_macs2_training_sequences.txt
#awk '{ print $2"\t"$3"\t"$4"\t"$5}' H1_ATAC_macs2_training.txt > H1_ATAC_macs2_training_numeric_scores.txt
#awk '{ print $NF}' H1_ATAC_macs2_training.txt | sed 's/0/\-1/g' > H1_ATAC_macs2_training_labels.txt
#
#awk '{ print $1}' H1_ATAC_macs2_test.txt > H1_ATAC_macs2_test_sequences.txt
#awk '{ print $2"\t"$3"\t"$4"\t"$5}' H1_ATAC_macs2_test.txt > H1_ATAC_macs2_test_numeric_scores.txt
#awk '{ print $NF}' H1_ATAC_macs2_test.txt | sed 's/0/\-1/g' > H1_ATAC_macs2_test_labels.txt
