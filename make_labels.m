function [ labels ] = make_labels(subject, label_type)
%inputs: subject object and the type of labels we want to extruct.
%(CL_level (task's level)/ ringSize/ condition/ nLevel (N-back))
%Optional: which data channel we want to extruct from subject to signals
%(Default: all)
%output: array of data signals corresponding to the blocks order. (Or: cell array of the signals arrays 
% for all channels) and array of labels by order with the type specified

len = size(subject.blocks); %Number of blocks need to be equal to length of channel

switch(label_type)
    case CL_level, %Need to calculate the level
        for i = 1:len
            block_level = level(subject.blocks(i).nLevel, subject.blocks(i).ringSize);
            labels(i) = block_level;
         end
    case condition,
        for i = 1:len
            labels(i) = subject.blocks(i).condition;
        end
    case nLevel,
        for i = 1:len
            labels(i) = subject.blocks(i).nLevel;
        end        
    case ringSize,
        for i = 1:len
            labels(i) = subject.blocks(i).ringSize;
        end

        
end

   
end


