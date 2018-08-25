%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%LOAD_DATA:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%This script loads all subjects files and arranges the data as inputs and labels to the
%network (in Python).
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%PARAMETERS:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Experiment's design:
    %# of blocks per condition: (Stress/ No Stress)
    global BLOCKS; 
    BLOCKS = 26;
    %# of difficulty levels: (include base level)
    % Change to 4 if want less labels
    global num_levels; %Including baseline
    num_levels = 6;
%%  
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Data's parameters:   
    %# of subjects:
    SUBJECTS = 25; 
    
    %Matrix of subjects' numbers that we won't load their files: 
    % Stress/ NoStress (ECG + GSR)
    filesNotes = 'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\fromYonit\data\notesBatData.xlsx';
    all_files = xlsread(filesNotes); %Read excel file with table which includes: 
    %index 1 to valid file and 0 to invalid.
    noStress_f = all_files(:,2)';
    Stress_f = all_files(:,3)';
    toInclude = [noStress_f; Stress_f];
    
    %toIclude(1,i) == 1 <=> take subject i noStress (ECG + GSR)
    %toIclude(2,i) == 1 <=> take subject i stress (ECG + GSR)
    
    %Matrix of subjects' numbers that update if their files were loaded successfully or not: 
    % Stress/ NoStress (ECG + GSR)
    %Initialize equal to toInclude
    filesLoaded = toInclude; 
    %filesLoaded(1,i) == 1 <=> subject i noStress were loaded (ECG + GSR)
    %filesLoaded(2,i) == 1 <=> subject i stress were loaded (ECG + GSR)
    
    
    %Number of valid subjects:
    valSUBJECTS = length(toInclude);
    
    %Number of channels (signals): ECG + GSR:    
    CHANNELS = 2; 
%%    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%For the network: inputes' parameters:
    %Train percentage:
    train_per = 0.7; 
    %window size: the input for network:
    seg_len = 2200; 
    %Over lap for the windows:
    over_lap = 0.2; 
    
    %RANDOM_ALL = true: if we want to split the data into train and test in random order
    %Which means we mix all the subjects'es data 
    %RANDOM_ALL = false: split subjects'es data into train or test separately
    RANDOM_ALL = true;
    
    %The number of subjects will use for train data if RANDOM_ALL == false
    %(train_per of the subjects number rounded)
    bound = -1;
    if RANDOM_ALL == false
        bound = round(valSUBJECTS*train_per);
    end
    
    %Make Stress/ No Stress labels
    make_StNoSt = 1; 
    
    %Random eack row in data also before calling divide_rand func
    rand_flag = 1;
    
    
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Files' parameters
    PATH = 'C:\Users\User\Documents\2017-2018\Project\MatlabScripts\load_data\fromYonit\data\'; 
    %Files' names:
    GSR_FILENAME_ST = '_BAT_ST_GSR'; 
    GSR_FILENAME_NS = '_BAT_NS_GSR';
    ECG_FILENAME_ST = '_BAT_ST_ECG';
    ECG_FILENAME_NS = '_BAT_NS_ECG';

    %**Isn't used for now..**
    COUNTER_SUB = zeros(SUBJECTS,BLOCKS*2); %COUNTER_SUB(i,j) == 1 
    % <=> subject i is loaded correctly and the data in the block j is in use
    % 1<=j<=BLOCKS : block j in 'no Stress' condition
    % BLOCKS+1<=j<=BLOCKS*2 : block j-BLOCKS in 'Stress' condition
    
    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%Make array of all subjects obj.
%For each subject:
    %Load the 4 files of data
    %Pre-processing the signals
    %Create subject obj. with: Blocks array (contains times etc.) and the signals
    %Add to subjects array subject object 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Loop for all subjects
j = 1;
for i = 1:SUBJECTS
        
    %Subject's number
    if i < 10
        sub_num = '00%d';
    elseif i >= 10
        sub_num = '0%d';
    end
    sub_num = sprintf(sub_num,i);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Load ECG files: stress + noStress:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if toInclude(1,i) == 1 %Load subject i files: NoStress
        
        %(NS)
        file_name = [PATH, sub_num, ECG_FILENAME_NS];
        file_name = join(file_name);
        try
            ecg_ns = load(file_name);
        catch
            filesLoaded(1,i) = 0; %Couldn't load the file
            fprintf('Couldnt load the %d ECG file NoStress\n',i);
            
        end
    end
    if toInclude(2,i) == 1 %Load subject i files: Stress
        %(ST)
        file_name = [PATH, sub_num, ECG_FILENAME_ST];
        file_name = join(file_name);
        try
            ecg_st = load(file_name);
        catch
            filesLoaded(2,i) = 0; %Couldn't load the file
            fprintf('Couldnt load the %d ECG file Stress\n',i);
        end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Filter the ECG data:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Remove Trends from Data
    %(https://www.mathworks.com/help/signal/ug/remove-trends-from-data.html)
    %%Eliminate linear trend
    %(NS)
    if filesLoaded(1,i) == 1 %Load subject i files: NoStress
        
        dt_ecgl = detrend(ecg_ns.EEG.data);
        ecg_ns.EEG.data = dt_ecgl;
    end
    %(ST)
    if filesLoaded(2,i) == 1 %Load subject i files: Stress
        dt_ecg2 = detrend(ecg_st.EEG.data);
        ecg_st.EEG.data = dt_ecg2;
    end
    %%Eliminate the nonlinear trend 
    opol = 6;
    %(NS)
    if filesLoaded(1,i) == 1 %Loaded subject i files: NoStress
        t1 = (1:length(ecg_ns.EEG.data));
        [p,s,mu] = polyfit(t1,ecg_ns.EEG.data,opol);
        f_y1 = polyval(p,t1,[],mu);
        dt_ecgnl = ecg_ns.EEG.data - f_y1;
        ecg_ns.EEG.data = dt_ecgnl;
    else %Not valid
        ecg_ns = '0';
    end
    %(ST)
    if filesLoaded(2,i) == 1 %Loaded subject i files: Stress
        t2 = (1:length(ecg_st.EEG.data));
        [p,s,mu] = polyfit(t2,ecg_st.EEG.data,opol);
        f_y2 = polyval(p,t2,[],mu);
        dt_ecgn2 = ecg_st.EEG.data - f_y2;
        ecg_st.EEG.data = dt_ecgn2;
    else %Not valid
        ecg_st = '0';
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Parse all the blocks' times and parameters:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(More details explained in make_blocks() function)
    blocks = make_blocks(ecg_ns, ecg_st, num_levels); 

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%Load GSR files: stress + noStress:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %(NS)
    if filesLoaded(1,i) == 1 %Load subject i files: NoStress
        file_name = [PATH, sub_num, GSR_FILENAME_NS]; 
        file_name = join(file_name);
        try
            gsr_ns = load(file_name);
        catch
            filesLoaded(1,i) = 0; %Couldn't load the file
            fprintf('Couldnt load the %d GSR file NoStress\n',i);
        end
    end
    %(ST)
    if filesLoaded(2,i) == 1 %Load subject i files: Stress
        file_name = [PATH, sub_num, GSR_FILENAME_ST];
        file_name = join(file_name);
        try
            gsr_st = load(file_name);
        catch
            filesLoaded(2,i) = 0; %Couldn't load the file
            fprintf('Couldnt load the %d GSR file NoStress\n',i);
        end
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Filter GSR data:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if filesLoaded(1,i) == 1 %Load subject i files: NoStress
        smoothdata(gsr_ns.data,'movmean',20); %**Maybe we want to change window size or add other filters
    end
    if filesLoaded(2,i) == 1 %Load subject i files: Stress
        smoothdata(gsr_st.data,'movmean',20);
    end
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Down GSR's sampling rate:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Takes GSR data
    %Down the sampling rate - From 1000 hz to 250 hz:
    %Add zeros to start for sync GSR with ECG data:
    %Take starting time of the first block:
    %(NS): 
    if filesLoaded(1,i) == 1 %Loaded subject i files: NoStress
        gsr_ns_data = gsr_ns.data; 
        gsr_ns_down = downsample(gsr_ns_data,4);
        start_t = blocks{1,1}(1,1);
        gsr_ns_down = [zeros(1,start_t-1), gsr_ns_down];
    else
        gsr_ns_down = '0';
    end
    %(ST):
    if filesLoaded(2,i) == 1 %Loaded subject i files: Stress
        gsr_st_data = gsr_st.data;
        gsr_st_down = downsample(gsr_st_data,4);      
        start_t = blocks{2,1}(1,1);
        gsr_st_down = [zeros(1,start_t-1), gsr_st_down];
    else
        gsr_st_down = '0';
    end


    %Takes GSR and ECG data and combine the two conditions per each
    %GSR: 
    %gsr_ns_down or '0' if not valid file 
    %gsr_st_down or '0' if not valid file
    gsr_data{1,1} = gsr_ns_down;
    gsr_data{2,1} = gsr_st_down;
    
    %ECG:
    %ecg_ns.EEG.data or '0' if not valid file 
    %ecg_st.EEG.data or '0' if not valid file
    if filesLoaded(1,i) == 1 %Loaded subject i files: NoStress
        ecg_data{1,1} = double(ecg_ns.EEG.data);
    else
        ecg_data{1,1} = '0';
    end
    if filesLoaded(2,i) == 1 %Loaded subject i files: Stress
        ecg_data{2,1} = double(ecg_st.EEG.data);
    else
        ecg_data{2,1} = '0';
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Make subject obj. and add to subjects array:
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if (filesLoaded(1,i) == 1)|| (filesLoaded(2,i) == 1) %Loaded subject i files: Stress/NoStress
        subjects(j) = subject(blocks, ecg_data, gsr_data);
        sub_ind(j) = i; %Subject's number
        j = j+1;     
    end
end
 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Arrange the data to the network:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Logical array: val_subjects(i) == 1 <=> we loaded files of Stress or
    %NoStress of subject i
%     val_subjects = filesLoaded(1,:)|filesLoaded(2,:);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %Split the data into segments and labels: 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Counting how many segments we have per level:
        counter = zeros(1,num_levels); 
        counter_st = zeros(1,2);
        
        %Matrix of the data's segments: (inputs). Row i = Level i-1 ;1<=i<=num_levels
        data = cell(num_levels,1); 
        
        %Matrix of the data's segments: (inputs). Row 1 = NoStress; row 2 = Stress
        StNoSt_data = cell(2,1); 

        %Matrix of the segments' labels. Row i = Level i-1 ;1<=i<=6
        labels = cell(num_levels,1); 
        
        %Matrix of the segments' labels: Stress/NoStress conditions
        StNoSt_labels = cell(2,1); 
        
    %Loop for all subjects in : subjects: 
    %Per each subject:
        %Call the func. 'Parser_levels' 
        %Get cell array of the data split into segments and labels 
        %Add to 'data' and 'labels'
    
    valSUBJECTS = (j-1); %Update number of valid subjects
    for i = 1:valSUBJECTS
        sub_num = sub_ind(i);
        [tmp_data, tmp_labels, tmp_counter, tmp_st_data, tmp_st_labels, tmp_counter_st ] = parser_levels(subjects(i), seg_len, over_lap, filesLoaded, sub_num,make_StNoSt, num_levels );
        
        
        %Update data and labels
        for row = 1:num_levels
            last = counter(1,row); %Number of segments we have so far 
            to_add = tmp_counter(1,row); %Number of segments we want to add 
            data(row, last+1: last + to_add) = tmp_data(row,1:to_add); %Add new segments in the row
            labels(row, last+1: last + to_add) = tmp_labels(row,1:to_add); %Add new segments in the row

        end
        
        if make_StNoSt == 1
            for row = 1:2
                last = counter_st(1,row);  %Number of segments we have so far 
                to_add = tmp_counter_st(1,row); %Number of segments we want to add 
                StNoSt_data(row, last+1: last + to_add) = tmp_st_data(row,1:to_add); %Add new segments in the row
                StNoSt_labels(row, last+1: last + to_add) = tmp_st_labels(row,1:to_add); %Add new segments in the row
            end
            %Update the global counter
            counter_st = counter_st + tmp_counter_st; 
        end
        
        %Update the global counter
        counter = counter + tmp_counter; 

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %Split the data and labels into test and train: 
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %***NEED TO UPDATE: NO WORKING FOR NOW
            %If we split the data in to test and train by subjects:
            %Means RANDOM_ALL == false
            %Train data and labels:
            if i == bound && RANDOM_ALL == false
                [train_input, train_labels] = make_inputs_labels(data, labels, counter); 
                counter = zeros(1,num_levels); %Reset the counter
                %**Need to reset data and labels also
            %Test data and labels:
            elseif  i == valSUBJECTS && RANDOM_ALL == false
                [test_input, test_labels] = make_inputs_labels(data, labels, counter); 
            end
    end
    
%Set the data that each level will have equal size of data: Random before between subjects if rand_flag == 1.     
[new_data, new_labels, new_counter ] = equal_size( data, labels, counter, rand_flag );   
 
data = new_data ;
labels = new_labels ;
counter = new_counter;


%If we split the data in to test and train not by subjects:
%Means RANDOM_ALL == true
if RANDOM_ALL == true
    %Split data and labels into train and test
    [train_input, train_labels, test_input, test_labels] = divide_rand(data, labels, counter, train_per); 
    if make_StNoSt == 1
        [train_input_st, train_labels_st, test_input_st, test_labels_st] = divide_rand(StNoSt_data, StNoSt_labels, counter_st, train_per); 
    end
end

%Added need to change
%[train_input_st, train_labels_st, test_input_st, test_labels_st] = classify(train_input_st, train_labels_st, test_input_st, test_labels_st, 1);

%%Make dataset with features
[dataset] = make_dataset(subjects);
%Encode Stress/NoStress 
T = strcmp(dataset.conditions, 'stress');
dataset.conditions = T;
dataset_n = dataset;
%Change the order of dataset
%dataset_n = [dataset(:,1:2),dataset(:,5:14),dataset(:,17:18),dataset(:,3)];
%Save to .csv file for learning
writetable(dataset_n,'data.csv') ;

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Save .mat file includes the test and train arrays:
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 save('mat_to_python.mat','train_input','train_labels','test_input','test_labels');
 if make_StNoSt == 1
    save('mat_to_python_st.mat','train_input_st','train_labels_st','test_input_st','test_labels_st');   
 end