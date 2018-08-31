function [features] = mat_features(signal, flag)
%This function gets ECGsignal, GSRsignal and returns features array as we
%used for the learning stage the X_input.
%flag = 0 <=> signal ecg. flag = 1 <=> signal gsr.

    if flag == 0
        ecg_data = signal;
        %ECG
        [avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= ECG_features(ecg_data);
        ecg_avgHR = avgHR;
        ecg_meanRR = meanRR;
        ecg_rmssd = rmssd;
        ecg_nn50 = nn50;
        ecg_pNN50 = pNN50;
        ecg_sd_RR = sd_RR;
        ecg_sd_HR = sd_HR;
        ecg_se = se;
        ecg_pse = pse;
        ecg_average_hrv = average_hrv;
        ecg_hrv = hrv; 
        features = table(ecg_avgHR ,ecg_meanRR ,ecg_rmssd ,ecg_nn50, ecg_pNN50,ecg_sd_RR, ecg_sd_HR,ecg_se,ecg_pse,ecg_average_hrv);
        
    else %flag == 1
        gsr_data = GSRsignal;
        %GSR
        gsr_mean = mean(gsr_data);
        gsr_std = std(gsr_data);
        [outNo,outMeanAmplitude,outMeanTime,outstdAmplitude,outstdTime,outminamplitude,outminTime,outmaxamplitude,outmaxTime,outmedianamplitude,outmedianTime] = feature_gsr(gsr_data);
        gsr_outNo = outNo;
        gsr_outMeanAmplitude = outMeanAmplitude;
        gsr_outMeanTime = outMeanTime;
        gsr_outstdAmplitude= outstdAmplitude;
        gsr_outstdTime = outstdTime;
        gsr_outminamplitude = outminamplitude;
        gsr_outminTime = outminTime;
        gsr_outmaxamplitude = outmaxamplitude;
        gsr_outmaxTime = outmaxTime;
        gsr_outmedianamplitude= outmedianamplitude;
        gsr_outmedianTime= outmedianTime;
        features = table(gsr_mean, gsr_std,gsr_outNo,gsr_outMeanAmplitude, gsr_outMeanTime,gsr_outstdAmplitude,gsr_outstdTime,gsr_outminamplitude,gsr_outminTime,gsr_outmaxamplitude,gsr_outmaxTime,gsr_outmedianamplitude,gsr_outmedianTime);%no raw signal


    end
        
  print('g')  
                    
    
%dataset = table(sub_num, conditions, ecg_signal, ecg_avgHR ,ecg_meanRR ,ecg_rmssd ,ecg_nn50, ecg_pNN50,ecg_sd_RR, ecg_sd_HR,ecg_se,ecg_pse,ecg_average_hrv, ecg_hrv, gsr_signal, gsr_mean, gsr_std, level);

end


