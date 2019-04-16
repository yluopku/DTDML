function [Theta_opt, obj_Phi] = optimizeTheta(H_Phi, vecY, Beta, Theta_ini, set, para)
% -------------------------------------------------------------------------
% Optimization of the combination coefficients of the bases
% -------------------------------------------------------------------------

maxit = 100;
sigma = para.sigma;
sigma_ = para.sigma;
epsilon = 1e-3;
Theta = zeros(size(Theta_ini)); % Theta = Theta_ini;
Theta_guess = zeros(size(Theta_ini)); % Theta_guess = Theta_ini;

obj_const = 0.5*(Beta'*(para.gammaA*set.preHB)*Beta);

fprintf('Optimizing Theta ... ');
starttime = cputime;
% -------------------------------------------------------------------------
% Initialization objective, gradient and Lipschitz constant
% -------------------------------------------------------------------------
[obj(1,1), grad(:,1), lipsc(1,1), obj_Phi, acc(1,1)] = ... 
    evaluate_cost(H_Phi, vecY, Beta, Theta, set, para, sigma, sigma_, obj_const);

% -------------------------------------------------------------------------
% Nesterov's optimal gradient method for TDML
% -------------------------------------------------------------------------
loop = 1; t = 1;
while loop
    % ------------------------------------------------------
    % Solving the two auxiliary optimization problems
    % ------------------------------------------------------
    y = Theta - 1.0/lipsc(t)*grad(:,t);
    tempGrad = zeros(size(grad,1), 1);
    for i = 1:t
        tempGrad = tempGrad + (i/2.0)*grad(:,i);
    end
    z = Theta_guess - 1.0/lipsc(t)*tempGrad;
    clear tempGrad
    
    % ------------------------------------------------------
    % Project the result to [0, 1]
    % ------------------------------------------------------
    % z = (z - min(z)) / (max(z) - min(z));
    
    % ------------------------------------------------------
    % Update the solution, objective, gradient and Lipschitz constant
    % ------------------------------------------------------
    Theta_new = (2.0/t+3)*z + (t+1)*1.0/(t+3)*y; clear y z
    t = t + 1;
    sigma = para.sigma / t;
    sigma_ = para.sigma / t;
    %     if sigma > 1
    %         sigma = para.sigma / t;
    %     end
    [obj(t,1), grad(:,t), lipsc(t,1), obj_Phi, acc(t,1)] = ... 
        evaluate_cost(H_Phi, vecY, Beta, Theta_new, set, para, sigma, sigma_, obj_const);
    
    % ------------------------------------------------------
    % Check convergence
    % ------------------------------------------------------
    if abs((obj(t,1) - obj(t-1,1))/(obj(t,1) - obj(1,1))) <= epsilon || t >= maxit
        loop = 0;
    end
    
    % ------------------------------------------------------
    % Updating Variables
    % ------------------------------------------------------
    if loop
        clear Theta
        Theta = Theta_new;
        clear Theta_new
    end
end
endtime = cputime;
fprintf('Finished! timecost = %.4f s \n', (endtime - starttime));
Theta_opt = Theta_new;

end

function [obj, grad, lipsc, obj_Phi, acc] = ... 
    evaluate_cost(H_Phi, vecY, Beta, Theta, set, para, sigma, sigma_, obj_const)
% -------------------------------------------------------------------------
% Compute the objective value, gradient and Lipschitz constant
% -------------------------------------------------------------------------

diagY = diag(vecY);
numer = vecY - diagY*(H_Phi'*Theta);
denom = sigma*max(H_Phi); denom = denom';
% denom = sigma*ones(set.nbPw, 1);

idx1 = find(numer > 0);
% idx2 = find(numer < -sigma);
idx2 = find(numer < -denom);
idx3 = setdiff((1:set.nbPw)', [idx1; idx2]);
acc = length(idx1)/set.nbPw;

idx1_ = find(Theta < -sigma_);
idx2_ = find(Theta > sigma_);
idx3_ = setdiff((1:set.nbBase)', [idx1_; idx2_]);

lipsc_Phi = max(1.0*(sum(H_Phi).^2) ./ denom') + para.gammaC / sigma_;
lipsc = lipsc_Phi + set.lipsc_Omega;

u = zeros(set.nbPw, 1);
u(idx2) = 1;
u(idx3) = - numer(idx3) ./ denom(idx3);
u_ = zeros(set.nbBase, 1);
u_(idx1_) = -1;
u_(idx2_) = 1;
u_(idx3_) = Theta(idx3_) ./ sigma_;
grad_Phi = 1.0/set.nbPw*(H_Phi*diagY*u) + para.gammaC*u_;
H_Omega = (para.eta + para.gammaA)*set.preHC;
h_Omega = para.gammaA*(set.preh*Beta);
grad_Omega = (H_Omega)*Theta - h_Omega;
grad = grad_Phi + grad_Omega;
clear tempH

obj_Phi_temp = zeros(set.nbPw, 1);
obj_Phi_temp(idx2) = - numer(idx2) - 0.5*denom(idx2);
obj_Phi_temp(idx3) = numer(idx3).^2 ./ (2.0*denom(idx3));
obj_Phi_ = zeros(set.nbBase, 1);
obj_Phi_(idx1_) = - Theta(idx1_) - sigma_ / 2;
obj_Phi_(idx2_) = Theta(idx2_) - sigma_ / 2;
obj_Phi_(idx3_) = Theta(idx3_).^2 / (2*sigma_);
obj_Phi = mean(obj_Phi_temp(:)) + para.gammaC*sum(obj_Phi_(:)); clear obj_Phi_temp obj_Phi_
obj_Omega = 0.5*(Theta'*(H_Omega)*Theta) ... 
    - Theta'*h_Omega;
obj = obj_Phi + obj_Omega;

obj = obj + obj_const;

end

