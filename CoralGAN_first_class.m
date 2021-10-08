clc;
clear;
load first_class;
xtrain = Real_first_class(1:90,:);
%% Ԥѵ��ģ��SAE
sae = saesetup([1251 300 50]);
opts.numepochs = 200;
opts.batchsize = size(xtrain,1);
sae = saetrain(sae, xtrain, opts);
%% ���Ա�����Ȩ�ظ���ͬ�ṹ��������
nn = nnsetup_nn([1251 300 50]);
nn.activation_function              = 'sigm';
nn.W{1} = sae.ae{1}.W{1};
nn.W{2} = sae.ae{2}.W{1};
nn.coral_ratio = 1;%����Ϊ1
nn_sou = nn;
nn_tar = nn;
%% GAN����
% -----------------����ģ��
generator = nnsetup_d([100, 128, 1251]);
discriminator = nnsetup_d([1251, 256, 1]);
%% ѵ������-----------��ʼѵ��
batch_size = 10;
epoch = 120;
data_num = 90;
batch_num = ceil(data_num / batch_size);%���ش��ڻ��ߵ���ָ�����ʽ����С����ͷ�ļ�
learning_rate = 0.001;
%% ѵ������
for e=1:epoch
    kk = randperm(data_num);
   for t=1:batch_num
        % ׼������
        data_real = xtrain(kk((t - 1) * batch_size + 1:t * batch_size),:);
        noise = unifrnd(-1, 1, batch_size,100);%unifrnd���ɱ�A��Bָ�����¶˵�[A,B]���������ȷֲ����������R
        % ��ʼѵ��
        % -----------����generator���̶�discriminator
        %����������
        generator = nnff(generator, noise);
        data_fake = generator.layers{generator.layers_count}.a;
        % ����������
        discriminator = nnff(discriminator, data_fake);
        logits_fake= discriminator.layers{discriminator.layers_count}.z;
        discriminator = nnbp_d(discriminator, logits_fake, ones(batch_size, 1));%��d�Ĳв�
        generator_d = nnbp_g(generator, discriminator);%��g�Ĳв�
        % coral����
        [nn_sou,nn_tar] = nnff_nn(nn_sou,nn_tar,data_real,data_fake);
        nn = nnbp_nn_gai(nn_sou, nn_tar, nn);%��nn�в� ��Ҫ�������nn�����Դ����� ǰ���ܵ�ʱ�������nn�����ԣ�Ȼ�������ֵ�����nn.����
        generator_nn = nnbp_nn_g(generator, nn); %��g�Ĳв� 
        % �ϲ�����
        generator = nn_d(generator_nn, generator_d, generator);%�ϲ��ݶ�,�������generator�����Դ�����
        generator = nnapplygrade(generator, learning_rate);
        % -----------����discriminator���̶�generator
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
%% ��������������
noise = unifrnd(-1, 1, 120,100);
generator = nnff(generator, noise);
Fake_first_class= generator.layers{generator.layers_count}.a;
