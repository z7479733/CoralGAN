function [nn_sou, nn_tar] = nnff_nn(nn_sou, nn_tar, x, x_tar)
% 返回各层的输出值、误差和损失函数(nn.a, nn.e and nn.L)
    n = nn_sou.n;
    
    m = size(x, 1);
    m_tar = size(x_tar, 1);          % 添加部分-计算目标域数据样本数
    x = [ones(m,1) x];
    x_tar = [ones(m_tar,1) x_tar];   % 添加部分-为目标域数据加一列元素1
    
    nn_sou.a{1} = x;
    nn_tar.a{1} = x_tar;             % 添加部分-将处理后的目标域数据作为输入

    % 前向传播(n-1是因为最后一层是softmax)
    %for i = 2 : n-1
    for i = 2 : n
        switch nn_sou.activation_function  % 假设源域和目标域激活函数相同！！！
            case 'sigm'
                % Calculate the unit's outputs (including the bias term)
                nn_sou.a{i} = sigm(nn_sou.a{i - 1} * nn_sou.W{i - 1}');
                nn_tar.a{i} = sigm(nn_tar.a{i - 1} * nn_sou.W{i - 1}');     % 添加部分-计算目标域数据在各层的输出
            case 'tanh_opt'
                nn_sou.a{i} = tanh_opt(nn_sou.a{i - 1} * nn_sou.W{i - 1}');
                nn_tar.a{i} = tanh_opt(nn_tar.a{i - 1} * nn_sou.W{i - 1}'); % 添加部分-计算目标域数据在各层的输出
        end
        
        if i == n
            nn_sou.feature_sou = nn_sou.a{i};        % 添加部分
            nn_tar.feature_tar = nn_tar.a{i};        % 添加部分
        end
       
        % 增加偏置项
        nn_sou.a{i} = [ones(m,1) nn_sou.a{i}];
        nn_tar.a{i} = [ones(m_tar,1) nn_tar.a{i}];   % 添加部分-为目标域数据添加偏置项
    end
    
%     switch nn_sou.output 
%         case 'sigm'
%             nn_sou.a{n} = sigm(nn_sou.a{n - 1} * nn_sou.W{n - 1}');
%         case 'linear'
%             nn_sou.a{n} = nn_sou.a{n - 1} * nn_sou.W{n - 1}';
%         case 'softmax'
%             nn_sou.a{n} = nn_sou.a{n - 1} * nn_sou.W{n - 1}';
%             nn_tar.a{n} = nn_tar.a{n - 1} * nn_sou.W{n - 1}';
%             nn_sou.a{n} = exp(bsxfun(@minus, nn_sou.a{n}, max(nn_sou.a{n},[],2)));
%             nn_sou.a{n} = bsxfun(@rdivide, nn_sou.a{n}, sum(nn_sou.a{n}, 2)); 
%     end

    % 误差与损失
%     nn_sou.e = y - nn_sou.a{n};
    
    % 计算CORAL-与省略部分计算结果相同
    nn_sou.d1 = size(nn_sou.feature_sou, 2);
    nn_tar.d1 = nn_sou.d1;
    nn_sou.Cor_sou = cov(nn_sou.feature_sou);%cov()方差
    nn_tar.Cor_tar = cov(nn_tar.feature_tar);
    
    nn_sou.Cor_tar = nn_tar.Cor_tar;
    nn_tar.Cor_sou = nn_sou.Cor_sou;
    
    nn_sou.Coral = sqrt(sum(sum((nn_sou.Cor_sou-nn_tar.Cor_tar).^2)))/(4 * nn_sou.d1 * nn_sou.d1);
    nn_sou.Coral = nn_sou.coral_ratio*nn_sou.Coral;
    nn_tar.Coral= nn_sou.Coral;
    
%     switch nn_sou.output
%         case {'sigm', 'linear'}
%             nn_sou.L = 1/2 * sum(sum(nn_sou.e .^ 2)) / m; 
%         case 'softmax'
%             nn_sou.L = -sum(sum(y .* log(nn_sou.a{n}))) / m  + nn_sou.Coral;
%     end
%      L(n) = nn_sou.L;%272个 从nnff_nn得到的L(n) 总的误差
%      Loss(n,1) = nn_sou.L - nn_sou.Coral; % 总的误差-coral误差=分类误差
%      Loss(n,2) = nn_sou.Coral; % coral误差
% 
%      n = n + 1;
end
