function sae = saesetup(size)
    for u = 2 : numel(size)
        sae.ae{u-1} = nnsetup_sae([size(u-1) size(u) size(u-1)]);
    end
end
