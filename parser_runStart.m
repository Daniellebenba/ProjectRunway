
function [ b ] = parser_runStart( EEG_event_code )
%Input: EEG.code : of runStart (ONLY) 
%Ouptput: Instance of the block with the assigned data

LEN = 21;
data = strsplit(EEG_event_code,'_'); %Split all the data string

if (size(data,2) == 20) %No Nback level
    
    values = [data(3),'-1'];    
    values = cat(2,values, data(6:2:20));  %Takes only the values
    
else %(size(data) == 21)
    values = data(3:2:LEN); %Takes only the values
    
end

b = block(values); %Create Block

end
