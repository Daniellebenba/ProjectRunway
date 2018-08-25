function [ inputs, inputs_labels ] = make_inputs_labels( data, labels, counter )
%Arrange data and labels into input and labels cell arrays 
%   
[m,num_of_labels] = size(counter);
counter_start = 0; % in order to concatinate data from all levels

for i = 2:num_of_labels %not including base-line    
    num_windows = counter(1,i);  
    counter_end = counter_start + num_windows;
    tmp_data = data(i,1:num_windows);
    tmp_labels = labels(i,1:num_windows);
    inputs(1,counter_start+1:counter_end)= tmp_data;
    inputs_labels(1,counter_start+1:counter_end)= tmp_labels;
    counter_start = counter_end;
   
end

  
end

