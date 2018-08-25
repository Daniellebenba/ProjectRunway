function [train_input , train_labels, test_input , test_labels] = divide_rand(data, labels, count, train_per)
%inputs: 
    %data: subject/subjects  cell arrays, each cell contains time-windowed
    %data matrixes of GSR and ECG according to level (0-5) 
    %[1,n] = size(data[i]), n = number of inputs of level i
    %labels: subject/subjects array containing arrays of labels of the data
    %count: an array, count.size = number of possible labels and
    %count[i] = number of inputs with label i
    %train_per = double between 0 and 1, the percentage of the data that
    %will be used for test. In each level data is devided according to train_per randomely 
%Returns:{train_input , train_labels, test_input , test_labels } each one
%is a cell array of size 1 X number of inputs (size(train_input) = size
%(train_labels), size(test_input) = size(test_labels)

[m,num_of_labels] = size(count); 

counter_train_start = 0; % in order to concatinate data from all levels
counter_test_start = 0;% in order to concatinate data from all levels

start = 1;
if num_of_labels == 2
    start = 1;
end
for i = start:num_of_labels %not including base-line    
    num_windows = count(1,i);
    P = train_per ;
    num_train_windows = round(P*num_windows);%number of inputs in training for level i
    num_test_windows = num_windows-num_train_windows;%number of inputs in training for level i  
    counter_train_end = counter_train_start+ num_train_windows;
    counter_test_end = counter_test_start+ num_test_windows;
    idx = randperm(num_windows)  ;
    tmp_data = data(i,:);
    tmp_labels = labels(i,:);
    train_input(counter_train_start+1:counter_train_end)= tmp_data(1,idx(1:round(P*num_windows))) ;
    test_input(counter_test_start+1:counter_test_end) = tmp_data(1,idx(round(P*num_windows)+1:end)) ;
    train_labels(counter_train_start+1:counter_train_end)= tmp_labels(1,idx(1:round(P*num_windows))) ;
    test_labels(counter_test_start+1:counter_test_end)= tmp_labels(1,idx(round(P*num_windows)+1:end)) ;
    counter_test_start = counter_test_end;
    counter_train_start = counter_train_end;
end

