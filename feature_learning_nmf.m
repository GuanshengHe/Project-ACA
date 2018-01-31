function [W, H]= feature_learning_nmf( V, order, num_iter )
% Use nonnegative matrix factorization to learn bases that best represent
% the input training set.
%
% Parameters
% ----------
% V : F x N array
% data matrix
% order : int
% vector dimension of each basis in the dictionary W
% num_iter : int
% number of iterations that NMF algorithm would perform
%
% Returns
% -------
% W : F x order array
% dictionary consisted of trained bases
% H : order x N array
% activation coefficients

% Initialize W and H randomly
F = size(V, 1);
N = size(V, 2);
W = (randi(51, F, order)-1) / 1000;
H = (randi(51, order, N)-1) / 1000;

% Perform NMF iteratively
for i_iter = 1 : num_iter
    
    %fprintf('NMF: Iteration %d.\n', i_iter)
    V_ = W*H;
    
    % Update W
    for a = 1 : order
        for i = 1 : F
            s =  sum(H(a, :).*V(i, :)./V_(i, :));
            W(i, a) = W(i, a) * s;
        end
        s = sum(W(:, a));
        W(:, a) = W(:, a) / s;
    end
    
    % Update H
    for a = 1 : order
        for u = 1 : N
            s = sum(W(:, a).*V(:, u)./V_(:, u));
            H(a, u) = H(a, u) * s;
        end
    end
end

end