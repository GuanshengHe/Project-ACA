function [ male_separated, female_separated ] = ...
    source_separation_alternative( mixed, window, overlap, ...
    W_male, W_female )
% Perform source separation on the mixed spectrogram V (alternative)
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
H = nmf_given_w(abs(V).^2, [W_male W_female]);

% Reconstruct two sources
order = size(H, 1) / 2;
spectrogram_male = W_male*H(1:order, :);
spectrogram_female = W_female*H(order+1:order*2, :);
phi = angle(V);
male_separated = ...
    ISTFT((spectrogram_male.^0.5).*exp(1i*phi), window, overlap);
male_separated = real(male_separated(1:length(mixed)));
female_separated = ...
    ISTFT((spectrogram_female.^0.5).*exp(1i*phi), window, overlap);
female_separated = real(female_separated(1:length(mixed)));

end

