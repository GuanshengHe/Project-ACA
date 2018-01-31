function V = load_traning_set( filepath, window, overlap )
% Load the training data set, compute the spectrogram of each data in the
% set and concatenate them into one matrix.
%
% Parameters
% ----------
% filepath : string
% full path to file with training data
% window : w x 1 array
% window used in STFT
% overlap : int
% number of overlapped samples between each window while performing STFT
%
% Returns
% -------
% V : nfft x N array
% concatenated spectrogram (N is the total number of spectrogram frames )

audio_list = dir([filepath '\*.wav']);
audio_list = struct2cell(audio_list)';
V = [];
for i = 1 : length(audio_list)
    [x, ~] = audioread([filepath '\' audio_list{i,1}]);
    X = STFT([x' zeros(1,(length(window)-overlap))]', window, overlap);
    V = [V abs(X).^2];
end

end

