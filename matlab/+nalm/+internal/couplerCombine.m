function [reflected, transmitted] = couplerCombine(cw, ccw, kappa)
%COUPLERCOMBINE Combine loop fields back into the linear arm and output.
%
%   The same 2x2 matrix as couplerSplit is used.  "reflected" corresponds to
%   the field that returns towards the linear arm.  "transmitted" corresponds
%   to the field launched into the output port.

    k = kappa;
    t = sqrt(max(0, 1 - k));
    r = 1i * sqrt(k);

    reflected = t * cw + r * ccw;
    transmitted = r * cw + t * ccw;
end
