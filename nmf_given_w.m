function H = nmf_given_w( V, W )
% Perform nonnegative matrix factorization given W.
%
% Parameters
% ----------
% V : F x N array
% data matrix
% W : F x order array
% dictionary consisted of trained bases
%
% Returns
% -------
% H : order x N array
% activation coefficients

N = size(V, 2);
order = size(W, 2);
max_iter = 200;
H = (randi(51, order, N)-1) / 1000;

for i = 1 : max_iter
    
    V_ = W*H;
    
    % Convergence Check (before maximum iteration)
    if max(max(abs(V-V_))) < min(min(V)) / 10
        break;
    end
    
    % Update H
    for a = 1 : order
        
        s2 = sum(W(:, a));
        for u = 1 : N
            s1 = sum(W(:, a).*V(:, u)./V_(:, u));
            scale = H(a, u) / s2;
            H(a, u) = H(a, u) + scale*(s1-s2);
        end
        
    end
    
end

end

