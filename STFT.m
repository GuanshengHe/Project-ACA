function [ X ] = STFT( x, window, overlap )
% Compute Short-Time Fourier Transform
%   x: time-domain signal
%   window: window used in the STFT
%   overlap: overlap between two windows

x = x(:);
window = window(:);

% Break the time domain signal into segments
x_t = buffer(x, length(window), overlap);

% Compute STFT
X = fft( x_t.*kron( window, ones(1,size(x_t, 2)) ));

end

