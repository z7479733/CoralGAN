function g_net = nnbp_nn_g(g_net, nn_net)
    n = g_net.layers_count;
    a = g_net.layers{n}.a;
    m = size(a,1);
    a = [ones(m,1) a];
    % generator��loss����label_fake�õ��ģ�(images_fake��discriminator�õ�label_fake)
    % ��g����bp��ʱ�򣬿��Խ�g��d������һ������
    % g���һ��Ĳв����d��2��Ĳв����(a .* (a_o))
    % g_net.layers{n}.d = nn_net.d{2} * nn_net.layers{2}.w' .* (a .* (1-a));
%      c=nn_net.d{2}(:,2:end) ;
%      d= nn_net.W{1};
    g_net.layers{n}.d = (nn_net.d{2}(:,2:end) * nn_net.W{1}) .* (a .* (1-a));
    g_net.layers{n}.d = g_net.layers{n}.d(:,2:end);
    for i = n-1:-1:2
        d = g_net.layers{i+1}.d;
        w = g_net.layers{i+1}.w;
        z = g_net.layers{i}.z;
        % ÿһ��Ĳв��Ƕ�ÿһ���δ����ֵ��ƫ�����������Ǻ�һ��Ĳв����w,�ٳ��϶Լ���ֵ��δ����ֵ��ƫ����
        g_net.layers{i}.d = d*w' .* delta_relu(z);    
    end
    % �������Ĳв�֮�󣬾Ϳ��Ը��ݲв��������loss��weights��biases��ƫ����
    for i = 2:n
        d = g_net.layers{i}.d;
        a = g_net.layers{i-1}.a;
        % dw�Ƕ�ÿ���weights����ƫ���������
        g_net.layers{i}.dw = a'*d / size(d, 1);
        g_net.layers{i}.db = mean(d, 1);%����d����ÿ�е�ƽ��ֵ
    end
end
