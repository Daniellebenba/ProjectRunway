function [ new_data, new_labels, new_counter ] = equal_size( data, labels, counter, rand_flag )
%This function takes data (num_levels)X(..) and labels (num_levels)X(..) cell array of data 
%(with all the subjects' %data, divide into segments with labels)
%Takes counter vector with count for each level row
%If rand_flag == 1 : Randomized the order of all subjects 
%Arrange the data and labels that will have equal size for each level
% new_data and new_labels size is (num_levels)X(min row size)

min_row = min(counter);

[n_rows,n_col] = size(data);

for row = 1:n_rows
    num_windows = counter(1,row);
    tmp_data = data(row,:);  %Copy the row from data
    tmp_labels = labels(row,:);  %Copy the row from labels
    if rand_flag == 1
        idx = randperm(num_windows); %Make randomized array of indexes
        tmp_data = tmp_data(1,idx); %Random the copied row
        tmp_labels = tmp_labels(1,idx); %Random the copied row
    end
    
    new_data(row,1:min_row) = tmp_data(1:min_row); %Add to new_data the new randomized row
    new_labels(row,1:min_row) = tmp_labels(1:min_row); %Add to new_data the new randomized row
end

new_counter =min_row*ones(1,n_rows);
