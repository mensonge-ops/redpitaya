function [u1o,u2o] = coupler(u1i,u2i,rho)
%COUPLER Fibre coupler matrix for complex field envelopes.
%   [U1O,U2O] = COUPLER(U1I,U2I,RHO) mixes the input fields U1I and U2I
%   according to the power splitting ratio RHO (0..1) using the standard
%   unitary 2x2 coupler matrix.

    rho = max(0,min(1,rho));
    u1o = sqrt(rho)*u1i + 1i*sqrt(1-rho)*u2i;
    u2o = 1i*sqrt(1-rho)*u1i + sqrt(rho)*u2i;
end
