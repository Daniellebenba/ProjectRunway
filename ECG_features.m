
% ecg_signal = dataset.ecg_signal{100,1}(1,1:500);

% [avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= ECG_Features(ecg_1)



%% ECG feature Extractor Toolbox
% Author: Shantanu V. Deshmukh
% Organization: University of Michigan Dearborn
% Version: 1.1

% This function computes the ECG features based on temporal as well as spectral analysis
% Note: 

% avgHR=Average Heart Rate
% meanRR= mean R-R interval distance 
% rmssd= Root Mean Square Distance of Successive R-R interval
% nn50= Number of R peaks in ECG that differ more than 50 millisecond
% pnn50= percentage NN50
% sd_RR= Standard Deviation of R-R series
% sd_HR= Standard Deviation of Heart Rate
% se= Sample Entropy
% pse= Power Spectral Entropy
% average_hrv= Average Heart Rate Variability
% hrv= Heart Rate Variability

% use:[avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= getECGFeatures(ecg_signal,winsize,wininc)


 function [avgHR,meanRR,rmssd,nn50,pNN50,sd_RR,sd_HR,se,pse,average_hrv,hrv]= ECG_Features(ecg_signal)
counter = 1;

%% Removing lower frequencies
    samplingrate=250;
    fresult=fft(ecg_signal);
    fresult(1 : round(length(fresult)*5/samplingrate))=0;
    fresult(end - round(length(fresult)*5/samplingrate) : end)=0;
    corrected=real(ifft(fresult));
    
    %% Filter - first pass
    WinSize = floor(samplingrate * 571 / 1000);
    if rem(WinSize,2)==0
        WinSize = WinSize+1;
    end
    filtered1=ecgdemowinmax(corrected, WinSize);
    
    % Scale an ecg
    peaks1=filtered1/(max(filtered1)/7);
    positions=find(peaks1);
    distance=positions(2)-positions(1);
    
    % Returns minimum distance between two peaks
    for data=1:1:length(positions)-1
        if positions(data+1)-positions(data)<distance 
            distance=positions(data+1)-positions(data);
        end
    end
    
    %% Optimize filter window size
    QRdistance=floor(0.04*samplingrate);
    if rem(QRdistance,2)==0
        QRdistance=QRdistance+1;
    end
    WinSize=2*distance-QRdistance;
    
    %% Filter - second pass
    filtered2=ecgdemowinmax(corrected, WinSize);
    peaks2=filtered2;
    
%% Plotting the HRV Heart Rate Variability
    [pks,locs]=find(peaks2);

     % Number of R-peaks 
     num_R_peaks=length(pks);
     % Calculating Average Heart Rate
     time_in_sec=(WinSize/samplingrate);
     averageHeartRate(counter)=(num_R_peaks/time_in_sec)*60;
     avgHR=averageHeartRate(counter);
     
     positions2=find(peaks2);
    for count=1:1:length(positions2)-1
        dstnce(count)=positions2(count+1)-positions2(count); % average the dstance to get average hrv in order to have one measure per window
        heartRate(count)=60*samplingrate/dstnce(count);
    end
    
    %Average HRV Calculation
    average_hrv=mean(dstnce);
    
%% HRV Parameters
% note RRinterval=dstnce array 
    %extracting Mean R-R interval
    mean_rr_interval=sum(dstnce)/length(dstnce);
    
    %Extracting Root Mean Square of the differences of successive R-R
    %interval    
    square_dstnce=dstnce.^2;
    avg_square_dstnce=sum(square_dstnce)/length(square_dstnce);
    rmssd_this=sqrt(avg_square_dstnce);
       
    %Extracting number of consecutive R-R intervals that differ more than
    %50 ms
    m=0;
    for num=1:1:length(dstnce)-1
        if(dstnce(num+1)-dstnce(num)<50)
            m=m+1;
        end
    end
    
    %Extracting Percentage value of total consecutive RR interval that
    %differ more than 50ms
    percentage_nn50= ((m(counter)/length(dstnce))*100);
    
    %Extracting Standard Deviation of RR interval series
    for num1=1:1:length(dstnce)
        sd1=mean_rr_interval-dstnce(num1);
    end
    sd2=sd1.^2;
    sd3=sum(sd2)/length(sd2);
    sd_final=sqrt(sd3);
    
    %Extracting Standard Dviation of Heart Rate
    for num2=1:1:length(heartRate)
        sd_heart1=averageHeartRate-heartRate(num2);
    end
    sd_heart2=sd_heart1.^2;
    sd_heart3=sum(sd_heart2)/length(sd_heart2);
    sd_heart_final=sqrt(sd_heart3);
    
    std_sig= std(ecg_signal);
    
%% Calculating Sample Entropy
    %I added new SampEn source code and decided to use corrected. All other
    %args were before
    sampleEntropy(counter) = SampEn(2 ,0.2*std_sig, ecg_signal, 1 );
    
    
%% Calculating Power Spectral density 

    % Calculating Power Spectral Entropy
    Fs1=250; %sampling rate per second
    Ts1=1/Fs1;%sampling time interval in second
    t1=0:Ts1:1-Ts1; n1=length(t1);

    fresult1=fft(dstnce);
    sum_fresult1=0.0;
    for i=1:1:length(fresult1)-1
     sum_fresult1= sum_fresult1 + (abs(fresult1(i)));
    end
    fresult1=fresult1/sum_fresult1;
    entropy1=0.0;
    for i=1:1:length(fresult1)-1
     entropy1= entropy1+ abs(fresult1(i))*log(1/abs(fresult1(i)));
     pse_rr=entropy1;
    end
    
    hrv=dstnce;
    meanRR=mean_rr_interval;
    rmssd=rmssd_this;
    nn50=m;
    pNN50=percentage_nn50;
    sd_RR= sd_final;  
    sd_HR=sd_heart_final;      
    se=sampleEntropy;
    pse=pse_rr;

 end

