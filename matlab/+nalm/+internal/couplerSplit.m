function [cw, ccw] = couplerSplit(in_main, in_aux, kappa)
%COUPLERSPLIT Split inputs into clockwise and counter-clockwise fields.
%
%   The 2x2 coupler matrix is defined as
%       [sqrt(1-k)    1i*sqrt(k)]
%       [1i*sqrt(k)   sqrt(1-k)]
%
%   in_main is the field entering the primary port (from the linear arm),
%   in_aux is the field entering the auxiliary port (usually vacuum).

    k = kappa;
    t = sqrt(max(0, 1 - k));
    r = 1i * sqrt(k);

    cw = t * in_main + r * in_aux;   % clockwise travelling field
    ccw = r * in_main + t * in_aux;  % counter-clockwise travelling field
end
