function [source, num_data] = load_testing_set( filepath )
% Load the testing data set and concatenate them into one matrix.
%
% Parameters
% ----------
% filepath : string
% full path to file with testing data
%
% Returns
% -------
% source : 1 x N cell
% set of testing data (N is the number of data in the set)
% num_data : int
% number of data in the testing data set

audio_list = dir([filepath '\*.wav']);
audio_list = struct2cell(audio_list)';
source = {};
num_data = length(audio_list);
for i = 1 : num_data
    [x, fs] = audioread([filepath '\' audio_list{i,1}]);
    source{i} = {x, fs};
end

end

