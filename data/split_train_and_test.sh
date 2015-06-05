shuf -n 100 iris.data  > iris_train
fgrep -x -f iris_train -v iris.data > iris_test
