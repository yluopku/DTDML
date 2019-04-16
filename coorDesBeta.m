function beta_new = coorDesBeta(H, h, beta, set, para)

if length(beta) <= 1
    beta_new = beta;
else
    loop = 1; iter = 0;
    obj = 0.5*(beta'*H*beta + para.gammaB*(beta'*beta)) - h'*beta;
    beta_new = beta;
    while loop
        iter = iter + 1;
        % --------------------------------------------
        % Randomly select two elments to update
        % --------------------------------------------
        %     [maxGrad, idxMax] = max(grad);
        %     [minGrad, idxMin] = min(grad);
        %     i = idxMax; j = idxMin;
        %     clear maxGrad idxMax minGrad idxMin
        % rand('seed', iter);
        betaPerm = randperm(set.nbSrc);
        for k = 1:1 % floor(set.nbSrc/2)
            i = betaPerm(2*k-1); j = betaPerm(2*k);
            % i = betaPerm(1); j = betaPerm(2);
            
            % --------------------------------------------
            % Update the selected two elments using coordinate descent
            % --------------------------------------------
            temp = H(i,i) - H(i,j) - H(j,i) + H(j,j);
            numer_i = para.gammaB*(beta(i)+beta(j)) + (h(i)-h(j)) + temp*beta(i) - (H(i,:)-H(j,:))*beta;
            numer_j = para.gammaB*(beta(i)+beta(j)) + (h(j)-h(i)) + temp*beta(j) - (H(j,:)-H(i,:))*beta;
            denom = temp + 2.0*para.gammaB; clear temp
            if numer_i <= 0
                beta_new(i) = 0; beta_new(j) = beta(i)+beta(j);
            end
            if numer_j <= 0
                beta_new(j) = 0; beta_new(i) = beta(i)+beta(j);
            end
            if numer_i > 0 && numer_j > 0
                beta_new(i) = numer_i / denom;
                beta_new(j) = beta(i) + beta(j) - beta_new(i);
            end
        end
        
        % --------------------------------------------
        % Check the convergence
        % --------------------------------------------
        obj_new = 0.5*(beta_new'*H*beta_new + para.gammaB*(beta_new'*beta_new)) - h'*beta_new;
        if abs(obj - obj_new) < 1e-4 || iter >= 500
            loop = 0;
        else
            clear beta
            beta = beta_new;
            obj = obj_new;
        end
    end
end

end

