classdef subject
        
    %  class of Subject. Subject contains blocks: array of block objects (with all the fields that
    %  can be used to labels.) and channels (ecg, gsr)
    
    properties
        blocks; %Array of blocks by order the subject participated
        channels; %Cell array of channels: ECG, GSR
        %**Need to add**Every channel has signals() and labels() (Every labels() is cell
        %array of different types of labels
        
    end
    
       %Constructor: gets blocks and ecg, gsr data
   methods
   function sub = subject(blocks, ecg_data, gsr_data)
       %if  nargin == 3 %If there are 3 arguments
       sub.blocks = blocks;
       sub.channels{1,1} = ecg_data;
       sub.channels{2,1} = gsr_data;
       %end
%        sub.channels(1).signals(), sub.channels(1).labels()  = {seg(ecg, block)};
%        sub.channels(2).signals() , sub.channels(2).labels() = {seg(ecg, block)};
              
   end
    
   %%**Needs to add* function that will make signals array with labels
   %%array (by type) by order
 
    
   end
end



