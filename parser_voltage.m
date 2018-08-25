%Takes data, start and end time of a block and returns array of the voltages from
%data
function [ voltages ] = parser_voltage(data, startTime, endTime )
  ind = startTime;
  index = ind - startTime +1;
  
    while (ind < endTime)
        voltages(1,index) = data(1,ind);
        ind = ind+1;
        index = index+1;
    end

end

