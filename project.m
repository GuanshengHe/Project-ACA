%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Main file of course project for A.C.A
% Created on Sun Apr 23 13:57 2017
% @author: Guansheng He
%%%%%%%%%%%%%%%%%%%%%%%%%%%%

close all
clear
clc


%% Load in training set and use NMF to learn the dictionary

% Define parameters
nfft = 1024;
overlap = nfft/2; % 50% overlap
w = @(n) (sin(pi/nfft*(n+0.5))); % N-point half-cycle sine window
order = 30;
num_iter = 50;

% Load training set of male and female speakers separately
if ~exist('.\data\dictionary\dictionary_30.mat', 'file')
    
    if exist('.\data\powerspectrum.mat', 'file')    
        disp('Loading training dataset...')
        load('.\data\powerspectrum.mat');
        disp('Finish loading.')
    
    else
        disp('Loading training set for male speakers...')
        training_male = load_traning_set('.\data\training\male', w(0:nfft-1)', overlap);
        disp('Finish loading.')
        
        disp('Loading traniing set for female speakers...')
        training_female = load_traning_set('.\data\training\female', w(0:nfft-1)', overlap);
        disp('Finish loading.')
        save('.\data\powerspectrum.mat', 'training_male', 'training_female', '-v7.3'); 
    end
    
    % Perform NMF on the data matrix (concatenated spectrogram)
    disp('Perform NMF on the male training dataset...')
    [W_male, ~] = feature_learning_nmf(training_male, order, num_iter);
    disp('Done.')
    disp('Perform NMF on the female training dataset...')
    [W_female, ~] = feature_learning_nmf(training_female, order, num_iter);
    disp('Done.')
    
    % Saving the dictionary
    disp('Save dictionary...')
    save('.\data\dictionary\dictionary_30.mat', 'W_male', 'W_female');
    disp('Done.')
    
end


%% Load in pre-learnt dictionary (if exists)
if exist('.\data\dictionary\dictionary_20.mat', 'file')
    load('.\data\dictionary\dictionary_20.mat', 'W_male', 'W_female')
    if ~exist('W_male', 'var')
        disp('Error: No pre-learnt dictionary exists!')
        order = 0;
    else
        order = size(W_male, 2);
    end
    
else
    disp('Error: No pre-learnt dictionary exists!')
    order = 0;
end


%% Load in testing dataset

% Load testing set of male and female speakers separately
disp('Loading testing set for male speakers...')
[testing_male, num_data_male] = load_testing_set('.\data\testing\male');
disp('Finish loading.')

disp('Loading testing set for female speakers...')
[testing_female, num_data_female] = load_testing_set('.\data\testing\female');
disp('Finish loading.')
    

%% Perform source separation on the whole testing set

if exist('testing_male', 'var') && exist('W_male', 'var')
    
    % Randomly select n samples from gender with larger set to mix with
    % sampes from gender with smaller set
    idx_male = 1:num_data_male;
    idx_male = idx_male(:, randperm(num_data_male));
    idx_female = 1:num_data_female;
    idx_female = idx_female(:, randperm(num_data_female));
    num_data = min(num_data_male, num_data_female);
    
    % Criteria for Method 1
    criteria_male = zeros(3, num_data);
    criteria_female = zeros(3, num_data);
    
    % Criteria for Method 2
    criteria_male1 = zeros(3, num_data);
    criteria_female1 = zeros(3, num_data);
    
    % Define parameters
    nfft = 1024;
    overlap = nfft/2; % 50% overlap
    w = @(n) (sin(pi/nfft*(n+0.5))); % N-point half-cycle sine window
    order_separate = 60;
    num_iter = 50;
    
    for idx = 1 : num_data
    
        male = testing_male{idx_male(idx)}{1};
        female = testing_female{idx_female(idx)}{1};
        data_len = min(length(male), length(female));
        mixed = male(1:data_len)+female(1:data_len);
        
        [male_separated, female_separated] = ...
            source_separation(mixed/max(mixed), w(0:nfft-1)', overlap, ...
            W_male, W_female, order_separate, num_iter);
        
        [male_separated1, female_separated1] = ...
            source_separation_alternative(mixed/max(mixed), ...
            w(0:nfft-1)', overlap, W_male, W_female);

        s = zeros(2,length(mixed));
        s(1,:) = male(1:length(mixed));
        s(2,:) = female(1:length(mixed));
        se = zeros(2,length(mixed));
        se(1,:) = male_separated;
        se(2,:) = female_separated;
        % Make sure the energy is not 0
        % (apply only when using the first method)
        se(:,1) = se(:,1) + eps;
        
        s1 = zeros(2,length(mixed));
        s1(1,:) = male(1:length(mixed));
        s1(2,:) = female(1:length(mixed));
        se1 = zeros(2,length(mixed));
        se1(1,:) = male_separated1;
        se1(2,:) = female_separated1;
        
        % Compare the separation result with pure source
        [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
        criteria_male(:, idx) = [SDR(1) SIR(1) SAR(1)];
        criteria_female(:, idx) = [SDR(2) SIR(2) SAR(2)];
        fprintf('Source separation: Sample %d.\n', idx)
        
        [SDR1, SIR1, SAR1, perm1] = bss_eval_sources(se1, s);
        criteria_male1(:, idx) = [SDR1(1) SIR1(1) SAR1(1)];
        criteria_female1(:, idx) = [SDR1(2) SIR1(2) SAR1(2)];
    end
    
    avg_male = sum(criteria_male, 2) / num_data;
    avg_female = sum(criteria_female, 2) / num_data;
    avg_male1 = sum(criteria_male1, 2) / num_data;
    avg_female1 = sum(criteria_female1, 2) / num_data;
    
else
    disp('Error: No enough variables.')   

end


%% Perform source separation on a single input
if exist('testing_male', 'var') && exist('W_male', 'var')
    
    % Randomly select n samples from gender with larger set to mix with
    % sampes from gender with smaller set
    idx_male = randi(num_data_male);
    idx_female = randi(num_data_female);
    
    % Define parameters
    nfft = 1024;
    overlap = nfft/2; % 50% overlap
    w = @(n) (sin(pi/nfft*(n+0.5))); % N-point half-cycle sine window
    order_separate = 20;
    num_iter = 50;
    
    male = testing_male{idx_male}{1};
    female = testing_female{idx_female}{1};
    data_len = min(length(male), length(female));
    mixed = male(1:data_len)+female(1:data_len);
    
    %[male_separated, female_separated] = ...
    %    source_separation(mixed/max(mixed), w(0:nfft-1)', overlap, ...
    %    W_male, W_female, order_separate, num_iter);
    
    [male_separated, female_separated] = ...
        source_separation_alternative(mixed/max(mixed), ...
        w(0:nfft-1)', overlap, W_male, W_female);
    
    s = zeros(2,length(mixed));
    s(1,:) = male(1:length(mixed));
    s(2,:) = female(1:length(mixed));
    se = zeros(2,length(mixed));
    se(1,:) = male_separated;
    se(2,:) = female_separated;
    
    % Compare the separation result with pure source
    [SDR, SIR, SAR, perm] = bss_eval_sources(se, s);
    
    figure (1)
    subplot(2,2,1); plot(0:data_len-1, s(1,:))
    title('Pure Source (Male)')
    subplot(2,2,2); plot(0:data_len-1, s(2,:))
    title('Pure Source (Female)')
    subplot(2,2,3); plot(0:data_len-1, se(1,:))
    title('Separated Signal (Male)')
    subplot(2,2,4); plot(0:data_len-1, se(2,:))
    title('Separated Signal (Female)')
    
else
    disp('Error: No enough variables.')   

end