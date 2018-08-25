function [ Blocks ] = make_blocks( EEG_NS, EEG_ST ,num_levels )
%EEG_NS : if file is loaded otherwise: EEG_NS == 0
%EEG_ST : if file is loaded otherwise: EEG_ST == 0
%num_levels number of levels: 6 (original), 3
%Returns: 2x2 cell array: Blocks{1:2,1} = 2 matrix with size 26x3  which contains :
%all bat blocks:
%column 1: start time; 
%column 2: end time; 
%column 3: level;
%The 2 matrix: 1 Line: NoStress condition, 2 Line: Stress condition 
%Blocks{1:2,2} = 26x1 matriz of Blocks objects with all the fields

%%Define: # of blocks
BLOCKS = 26;

%%Creates subject. Gets ECG and GSR files (ST and NS) and returns subject object
%function subject =  create_subject(EEG_NS, EEG_ST, acq_NS, acq_ST)
 
Blocks{1,1} = zeros(BLOCKS, 3); %NoStress
Blocks{2,1} = zeros(BLOCKS, 3); %Stress

for j = 1:2 %Extruct from stress and no stress conditions

    index = 1; 
    foundStart = false; %Flag: if we found runStart we need to find the next runEnd 

    if (j == 1) && (isequaln(EEG_NS,'0')==0) %NoStress and file is valid
        EEG = EEG_NS;
    elseif (j == 1) && (isequaln(EEG_NS,'0')==1) %NoStress and file isn't valid so break
        continue;
    elseif (j == 2) && (isequaln(EEG_ST,'0')==0) %Stress and file is valid
        EEG = EEG_ST;
    elseif (j == 2) && (isequaln(EEG_ST,'0')==1)  %Stress and file isn't valid so break
        continue;
    end
    
    %Loop for making a matrix of the blocks and parse all block's info (nLevel, ringSize etc.)
    for t = EEG.EEG.event(1,:)
        if strcmp(t.type,'runStart') %Start of Block  
            Blocks{j,2}(index,1) =  parser_runStart(t.code);  %Array of all blocks objects
            Blocks{j}(index,1) = t.latency; %Start time of the block
            Blocks{j}(index, 3) = level(Blocks{j,2}(index,1).nLevel, Blocks{j,2}(index,1).ringSize, num_levels); %Calculate the block's level 
            foundStart = true; %Now we need to find runEnd of that block
        elseif strcmp(t.type, 'runEnd') && (foundStart) %Found start of a block and this is the block's end time
            Blocks{j}(index, 2) = t.latency;  %Block's end time
            index = index + 1; %Search for next block
        end
    end

end
% if j == 1
%     index = 1;
% else
%     index = BLOCKS+1;
% end

% %Make table of all blocks array of ecg voltages. row i = block i.
% for i = index:index+BLOCKS-1
%     start = Blocks(i,1);
%     endTime = Blocks(i,2);
%     ind = endTime - start ;
%     ecg_voltages(i,1:ind) = parser_voltage(EEG.data, start, endTime);
%     gsr_voltages(i,1:ind) = parser_voltage(acq.data(:,3), start, endTime);
%     **Needs to make gsr_voltages
% end
% 
% end
% 
% B_array = transpose(B_array);
% 
% %Create a class object of subject 
% subject = subject(B_array, ecg_voltages, gsr_voltages);


%end

end

