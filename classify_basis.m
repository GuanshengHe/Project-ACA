function label = classify_basis( W_male, W_female, basis )
% Using nearest-neighbor to classify the input basis
%
% Parameters
% ----------
% W_male : F x r array
% dictionary containing bases representing male speaker
% (F is number of bases and r is the vector dimension of each basis)
% W_female : F x r array
% dictionary containing bases representing female speaker
% basis : F x 1 array
% current basis waiting to be classified
%
% Returns
% -------
% label : int
% label that indicates the classifying result of the current basis
% (1 for male, 2 for female, 3 for basis existing in both dictionary)

% Calculate the squared difference between the current basis and each basis
% in two dictionaries
r = size(W_male, 2);
d_male = (W_male - basis(:, ones(r,1))).^2;
d_female = (W_female - basis(:, ones(r,1))).^2;

if min(sum(d_male, 1)) == min(sum(d_female, 1))
    label = 3;
else
    if min(sum(d_male, 1)) < min(sum(d_female, 1))
        label = 1;
    else
        label = 2;
    end
end

end