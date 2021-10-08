function nn = nnsetup_d(architecture)
    nn.architecture   = architecture;
    nn.layers_count = numel(nn.architecture);
    % t,beta1,beta2,epsilon,nn.layers{i}.w_m,nn.layers{i}.w_v,nn.layers{i}.b_m,nn.layers{i}.b_v��Ӧ��adam�㷨������������ı���
    nn.t = 0;
    nn.beta1 = 0.9;
    nn.beta2 = 0.999;
    nn.epsilon = 10^(-8);
    % ����ṹΪ[100, 512, 784]������3�㣬�����100���������ز㣺100*512��512*784, ���Ϊ���һ���aֵ������ֵ��
    for i = 2 : nn.layers_count   
        nn.layers{i}.w = normrnd(0, 0.02, nn.architecture(i-1), nn.architecture(i)); %��̬�ֲ���ʼ��Ȩ��
        nn.layers{i}.b = normrnd(0, 0.02, 1, nn.architecture(i));
        nn.layers{i}.w_m = 0;
        nn.layers{i}.w_v = 0;%��ʼ��Ϊ0
        nn.layers{i}.b_m = 0;
        nn.layers{i}.b_v = 0;
    end
end

