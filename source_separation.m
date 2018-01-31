function [ male_separated, female_separated ] = ...
    source_separation( mixed, window, overlap, ...
    W_male, W_female, order, num_iter )
% Perform source separation on the mixed spectrogram V
%
% Parameters
% ----------
% mixed : nsample x 1 array
% mixed audio
% window : nfft x 1 array
% window function used in STFT
% overlap : int
% number of samples overlapped of each window in STFT 
% W_male : F x r array
% dictionary containing bases representing male speaker
% (F is number of bases and r is the vector dimension of each basis)
% W_female : F x r array
% dictionary containing bases representing female speaker
% order : int
% vector dimension of each basis
% num_iter : int
% number of iterations that NMF algorithm would perform
%
% Returns
% -------
% male_separated : nsample x 1 array
% separated audio signal for male speaker
% female_separated : nsample x 1 array
% separated audio signal for female speaker

% Compute STFT
mixed = mixed(:);
nfft = length(window);
V = STFT([mixed' zeros(1,(nfft-overlap))]', window, overlap);

% Perform NMF on the mixed audio input
[W_test, H] = feature_learning_nmf(abs(V).^2, order, num_iter);

% Use nearest-neighbor to classify each basis
spectrogram_male = zeros(size(V));
spectrogram_female = zeros(size(V));
N = size(W_test, 2);
for i = 1 : N
    
    b = W_test(:, i);
    label = classify_basis(W_male, W_female, b);
    if (label == 1 || label == 3)
        spectrogram_male = spectrogram_male + b * H(i, :);
    end
    if (label == 2 || label == 3)
        spectrogram_female = spectrogram_female + b * H(i, :);
    end
    
end

% Reconstruct two sources
phi = angle(V);
male_separated = ...
    ISTFT((spectrogram_male.^0.5).*exp(1i*phi), window, overlap);
male_separated = real(male_separated(1:length(mixed)));
female_separated = ...
    ISTFT((spectrogram_female.^0.5).*exp(1i*phi), window, overlap);
female_separated = real(female_separated(1:length(mixed)));

end