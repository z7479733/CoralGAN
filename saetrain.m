function sae = saetrain(sae, x_data, opts)
    for i = 1 : numel(sae.ae)
        disp(['Training AE ' num2str(i) '/' num2str(numel(sae.ae))]);
        sae.ae{i} = nntrain_sae(sae.ae{i}, x_data, x_data, opts);
        t = nnff_sae(sae.ae{i}, x_data, x_data);
        x_data = t.a{2};
        % remove bias term
        x_data = x_data(:,2:end);
    end
end
