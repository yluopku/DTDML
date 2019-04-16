function Beta_new = optimizeBeta(Beta, Theta, set, para)
% -------------------------------------------------------------------------
% Optimization of the combination coefficients
% -------------------------------------------------------------------------

h = para.gammaA*(set.preh'*Theta);
H = para.gammaA*set.preHB;

Beta_new = coorDesBeta(H, h, Beta, set, para);

% matH = H + para.gammaB*eye(size(H));
% vech = -h;
% 
% lb = zeros(set.nbSrc, 1);
% ub = ones(set.nbSrc, 1);
% Beta_new = quadprog(matH, vech, [], [], [], [], lb, ub);

end

