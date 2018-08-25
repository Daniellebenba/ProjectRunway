%%Class of block with all the fields: 

classdef block
   properties
    condition; %stress/ noStress
    nLevel; %0,...7
    ringSize; %
    blockOrdinal;
    isPractice;
    blockNumber; %
    speed;
    subjectNumber; %
    isBaseline
    order;
   end
   
   %Constructor
   methods
   function b = block(values)
    b.condition = values(1); %stress/ noStress
    b.nLevel = values(2); %0,...7
    b.ringSize = values(3); %
    b.blockOrdinal = values(4);
    b.isPractice = values(5);
    b.blockNumber = values(6); %
    b.speed = values(7);
    b.subjectNumber = values(8); %
    b.isBaseline = values(9);
    b.order = values(10);
     
   end 
end
end

