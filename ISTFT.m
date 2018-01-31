function [ x ] = ISTFT( X, window, overlap )
% Inverse Short-Time Fourier Transform
%   X: STFT coefficients
%   window: window used in the inverse STFT
%   overlap: overlap between two windows

% Compute inverse Fourier transform of each block
x_t = ifft(X);

% Multiply each block by the window
win_size = length(window);
x_t = x_t(1:win_size, :).*kron( window, ones(1,size(x_t, 2)) );

% Reconstruct the signal by adding those overlapping blocks
l = size(x_t,1)*size(x_t,2)-overlap*(size(x_t,2)-1);
x = zeros(l,1);
for i = 1:size(x_t, 2)
    s = (i-1)*(win_size-overlap)+1;
    e = s + win_size-1;
    x(s:e) = x(s:e)+x_t(:,i);
end

x = x(overlap+1:l);

end

