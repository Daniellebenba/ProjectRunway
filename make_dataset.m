%function [ dataset ] = make_dataset( subjects, seg_len, over_lap )
%Takes subjects array and makes a dataest table

seg_len = 2200;
over_lap = 0.3;

ind = 1;
max_hrv = 0;
for sub = 1:length(subjects)
    
    sub_number = subjects(1,sub).blocks{1,2}(1,1).subjectNumber{1,1} %Subject's number
    sub_number = str2num(sub_number)
    for cond = 1:2
        [nr, nc] = size(subjects(1,sub).blocks{cond,2}); %Number of rows and columns in blocks in condition = cond.
        %condition = subjects(1,sub).blocks{1,2}(1,1).condition;
        condition = subjects(1,sub).blocks{cond,2}(1,1).condition;
        ecg_data = subjects(1,sub).channels{1,1}{cond,1};
        gsr_data = subjects(1,sub).channels{2,1}{cond,1};
        for b = 1:nr %Iterate all blocks in condition
            startTime = subjects(1,sub).blocks{cond,1}(b,1);
            endTime = subjects(1,sub).blocks{cond,1}(b,2);  
            %Take all the signal of block b: 
            %ECG
            [signal_e] = parser_voltage(ecg_data, startTime, endTime );
            %GSR
            [signal_g] = parser_voltage(gsr_data, startTime, endTime );
            %Split into segments if seg_len != -1
            if seg_len > -1 %Take windows not all block as input 
                %Make segments from the current block
                %index = startTime;
                index = 1;
                endTime = length(signal_e);
                back = seg_len*over_lap;
                while index + seg_len - 1 <= endTime 
                    
                    sub_num(ind,1) = sub_number; %Subject's number
                    conditions(ind,1) = condition; %Stress/ NoStress
                    level(ind,1) = subjects(1,sub).blocks{cond,1}(b,3); %Block's level                    
                    
                    %ECG
                    seg_ecg = signal_e(1,index:index+seg_len-1);
                    ecg_signal{ind,1} = [seg_ecg];                  

                    [avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= ECG_features(seg_ecg);
                    ecg_avgHR(ind,1) = avgHR;
                    ecg_meanRR(ind,1) = meanRR;
                    ecg_rmssd(ind,1) = rmssd;
                    ecg_nn50(ind,1) = nn50;
                    ecg_pNN50(ind,1) = pNN50;
                    ecg_sd_RR(ind,1) = sd_RR;
                    ecg_sd_HR(ind,1) = sd_HR;
                    ecg_se(ind,1) = se;
                    ecg_pse(ind,1) = pse;
                    ecg_average_hrv(ind,1) = average_hrv;
                    ecg_hrv{ind,1} = hrv;             
                    
                    %GSR
                    seg_gsr =  signal_g(index:index+seg_len-1);
                    gsr_signal{ind,1} = seg_gsr;
                    gsr_mean(ind,1) = mean(seg_gsr);
                    gsr_std(ind,1) = std(seg_gsr);
                    %Jump to next segment's starting index
                    index = index + seg_len - back;
                    ind = ind + 1;
                end
            else
                sub_num(ind,1) = sub_number; %Subject's number
                conditions(ind,1) = condition; %Stress/ NoStress
                level(ind,1) = subjects(1,sub).blocks{cond,1}(b,3); %Block's level 
                ecg_signal{ind,1} = signal_e;
                %Adding features of ECG signal:
                ecg_mean(ind,1) = mean(signal);
                ecg_std(ind,1) = std(signal);
    %           ecg_peaks{ind,1} = findpeaks(signal);
                [avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= ECG_features(signal_e);
                ecg_avgHR(ind,1) = avgHR;
                ecg_meanRR(ind,1) = meanRR;
                ecg_rmssd(ind,1) = rmssd;
                ecg_nn50(ind,1) = nn50;
                ecg_pNN50(ind,1) = pNN50;
                ecg_sd_RR(ind,1) = sd_RR;
                ecg_sd_HR(ind,1) = sd_HR;
                ecg_se(ind,1) = se;
                ecg_pse(ind,1) = pse;
                ecg_average_hrv(ind,1) = average_hrv;
                ecg_hrv{ind,1} = hrv;
%                 if max_hrv < length(hrv)
%                     max_hev = length(hrv);
%                 end
                %GSR
                gsr_signal{ind,1} = signal_g;
                %Adding features of GSR signal:
                gsr_mean(ind,1) = mean(signal);
                gsr_std(ind,1) = std(signal);
                ind = ind + 1;

        end
    end
    end
end
%Add zeros to fix length
% for i = 1:ind-1
%     ecg_hrv{i,1}(1,end+1:max_hev) = 0;
% end
%%If want to add feature: need to add in the args after: ecg_signal and gsr_signal
dataset = table(sub_num, conditions, ecg_signal, ecg_avgHR ,ecg_meanRR ,ecg_rmssd ,ecg_nn50, ecg_pNN50,ecg_sd_RR, ecg_sd_HR,ecg_se,ecg_pse,ecg_average_hrv, ecg_hrv, gsr_signal, gsr_mean, gsr_std, level);


