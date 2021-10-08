function nn = nnbp_nn(nn_sou,nn_tar)
    % ���練�򴫲������ظ��º��Ȩ�� 
   n = nn_sou.n+1;

    
    % ��Ӳ���-���㷴�����
    % Դ��
    size_sou = size(nn_sou.feature_sou, 1);
    size_tar = size(nn_tar.feature_tar, 1);
    
    d_cor_sou = (1/(nn_sou.d1^2*(size_sou-1))) * ((nn_sou.feature_sou'-(ones(size_sou,1)'*nn_sou.feature_sou)'*ones(size_sou,1)'/size_sou)'*(nn_sou.Cor_sou-nn_sou.Cor_tar));
    d_cor_tar = (-1/(nn_tar.d1^2*(size_tar-1))) * ((nn_tar.feature_tar'-(ones(size_tar,1)'*nn_tar.feature_tar)'*ones(size_tar,1)'/size_tar)'*(nn_tar.Cor_sou-nn_tar.Cor_tar));
   
    d_sou{n} = [ones(size_sou,1) nn_sou.coral_ratio*d_cor_sou];
    d_tar{n} = [ones(size_tar,1) nn_tar.coral_ratio*d_cor_tar];

    for i = (n-1) : -1 : 2 
        % ���㼤����ĵ���
        switch nn_sou.activation_function % ����Դ���Ŀ���򼤻����ͬ������
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
            d_tar{i} = d_tar{i+1}.* d_act_tar; % �˴��ݶȲ��Ӹ���
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
%         if i+1==n     % ������һ��͵����ڶ���֮��Ȩ�صĸ���
%             nn_sou.dW{i} = (d_sou{i + 1}' * nn_sou.a{i}) / size(d_sou{i + 1}, 1);
%             nn_tar.dW{i} = (d_tar{i + 1}' * nn_tar.a{i}) / size(d_tar{i + 1}, 1);
%         else          % ���������Ȩ�صĸ���
%             nn_sou.dW{i} = (d_sou{i + 1}(:,2:end)' * nn_sou.a{i}) / size(d_sou{i + 1}, 1); 
%             nn_tar.dW{i} = (d_tar{i + 1}(:,2:end)' * nn_tar.a{i}) / size(d_tar{i + 1}, 1);      
%         end
%     end
%    for j = 2:(n - 1)
%      nn.d{j} = d_sou{j} + d_tar{j};
%    end

