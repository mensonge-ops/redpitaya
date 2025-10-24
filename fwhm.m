function [width,I_l,I_r] = fwhm(x)
%FWHM Compute the discrete full width at half maximum of vector X.

    [peak, ind_peak] = max(x);
    half_peak = peak/2;
    x_l = x(1:ind_peak);
    x_r = x(ind_peak:end);
    I_l_rel = find(fliplr(x_l) <= half_peak, 1, 'first');
    I_r_rel = find(x_r <= half_peak, 1, 'first');
    width = I_l_rel + I_r_rel;
    I_l = ind_peak - I_l_rel;
    I_r = ind_peak + I_r_rel - 1;
end
