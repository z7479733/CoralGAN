function nn = nnbp_nn(nn_sou,nn_tar)
    % 网络反向传播，返回更新后的权重 
   n = nn_sou.n+1;

    
    % 添加部分-计算反向误差
    % 源域
    size_sou = size(nn_sou.feature_sou, 1);
    size_tar = size(nn_tar.feature_tar, 1);
    
    d_cor_sou = (1/(nn_sou.d1^2*(size_sou-1))) * ((nn_sou.feature_sou'-(ones(size_sou,1)'*nn_sou.feature_sou)'*ones(size_sou,1)'/size_sou)'*(nn_sou.Cor_sou-nn_sou.Cor_tar));
    d_cor_tar = (-1/(nn_tar.d1^2*(size_tar-1))) * ((nn_tar.feature_tar'-(ones(size_tar,1)'*nn_tar.feature_tar)'*ones(size_tar,1)'/size_tar)'*(nn_tar.Cor_sou-nn_tar.Cor_tar));
   
    d_sou{n} = [ones(size_sou,1) nn_sou.coral_ratio*d_cor_sou];
    d_tar{n} = [ones(size_tar,1) nn_tar.coral_ratio*d_cor_tar];

    for i = (n-1) : -1 : 2 
        % 计算激活函数的导数
        switch nn_sou.activation_function % 假设源域和目标域激活函数相同！！！
            case 'sigm'
                d_act_sou = nn_sou.a{i} .* (1 - nn_sou.a{i});
                d_act_tar = nn_tar.a{i} .* (1 - nn_tar.a{i});
            case 'tanh_opt'
                d_act_sou = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_sou.a{i}.^2);
                d_act_tar = 1.7159 * 2/3 * (1 - 1/(1.7159)^2 * nn_tar.a{i}.^2);
        end
        
        % Backpropagate first derivatives
         if i == n-1
            % in this case in d{n} there is not the bias term to be removed 
            d_sou{i} = d_sou{i+1}.* d_act_sou;
            d_tar{i} = d_tar{i+1}.* d_act_tar; % 此处梯度不加负号
            % d_tar{i} = (d_tar{i + 1} * nn.W{i} + sparsityError) .* d_act;   
        else   % in this case in d{i} the bias term has to be removed
            d_sou{i} = (d_sou{i + 1}(:,2:end) * nn_sou.W{i}) .* d_act_sou;
            d_tar{i} = (d_tar{i + 1}(:,2:end) * nn_tar.W{i}) .* d_act_tar;
        end
        nn.d{i} = d_sou{i} + d_tar{i};
%         nn.W{i-1} = nn_sou.W{i-1};
    end
end
%     for i = 1 : (n - 1)
%         if i+1==n     % 倒数第一层和倒数第二层之间权重的更新
%             nn_sou.dW{i} = (d_sou{i + 1}' * nn_sou.a{i}) / size(d_sou{i + 1}, 1);
%             nn_tar.dW{i} = (d_tar{i + 1}' * nn_tar.a{i}) / size(d_tar{i + 1}, 1);
%         else          % 其余网络层权重的更新
%             nn_sou.dW{i} = (d_sou{i + 1}(:,2:end)' * nn_sou.a{i}) / size(d_sou{i + 1}, 1); 
%             nn_tar.dW{i} = (d_tar{i + 1}(:,2:end)' * nn_tar.a{i}) / size(d_tar{i + 1}, 1);      
%         end
%     end
%    for j = 2:(n - 1)
%      nn.d{j} = d_sou{j} + d_tar{j};
%    end

