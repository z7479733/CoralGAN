clc;
clear;
load first_class;
xtrain = Real_first_class(1:90,:);
%% 预训练模型SAE
sae = saesetup([1251 300 50]);
opts.numepochs = 200;
opts.batchsize = size(xtrain,1);
sae = saetrain(sae, xtrain, opts);
%% 将自编码器权重赋给同结构的神经网络
nn = nnsetup_nn([1251 300 50]);
nn.activation_function              = 'sigm';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.coral_ratio = 1;%设置为1
nn_sou = nn;
nn_tar = nn;
%% GAN部分
% -----------------定义模型
generator = nnsetup_d([100, 128, 1251]);
discriminator = nnsetup_d([1251, 256, 1]);
%% 训练参数-----------开始训练
batch_size = 10;
epoch = 120;
data_num = 90;
batch_num = ceil(data_num / batch_size);%返回大于或者等于指定表达式的最小整数头文件
learning_rate = 0.001;
%% 训练部分
for e=1:epoch
    kk = randperm(data_num);
   for t=1:batch_num
        % 准备数据
        data_real = xtrain(kk((t - 1) * batch_size + 1:t * batch_size),:);
        noise = unifrnd(-1, 1, batch_size,100);%unifrnd生成被A和B指定上下端点[A,B]的连续均匀分布的随机数组R
        % 开始训练
        % -----------更新generator，固定discriminator
        %生成器部分
        generator = nnff(generator, noise);
        data_fake = generator.layers{generator.layers_count}.a;
        % 鉴别器部分
        discriminator = nnff(discriminator, data_fake);
        logits_fake= discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));%求d的残差
        generator_d = nnbp_g(generator, discriminator);%求g的残差
        % coral部分
        [nn_sou,nn_tar] = nnff_nn(nn_sou,nn_tar,data_real,data_fake);
        nn = nnbp_nn_gai(nn_sou, nn_tar, nn);%求nn残差 需要把上面的nn的属性传过来 前面跑的时候会生成nn的属性，然后这个赋值会添加nn.属性
        generator_nn = nnbp_nn_g(generator, nn); %求g的残差 
        % 合并部分
        generator = nn_d(generator_nn, generator_d, generator);%合并梯度,把上面的generator的属性传过来
        generator = nnapplygrade(generator, learning_rate);
        % -----------更新discriminator，固定generator
        generator = nnff(generator, noise);
        data_fake = generator.layers{generator.layers_count}.a;
        data = [data_fake;data_real];
        discriminator = nnff(discriminator, data);
        logits = discriminator.layers{discriminator.layers_count}.z;
        labels = [zeros(batch_size,1);ones(batch_size,1)];
        discriminator = nnbp_d(discriminator, logits, labels);
        discriminator = nnapplygrade(discriminator, learning_rate); 
   end
end
%% 生成器生成数据
noise = unifrnd(-1, 1, 120,100);
generator = nnff(generator, noise);
Fake_first_class= generator.layers{generator.layers_count}.a;
