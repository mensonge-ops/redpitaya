function tf = filter_lorentz_tf(ui,fbw,fc,fo,df)
%FILTER_LORENTZ_TF Return the normalised Lorentzian gain transfer function.
%   TF = FILTER_LORENTZ_TF(UI,FBW,FC,FO,DF) computes the Lorentzian spectral
%   profile centred at FC with full-width FBW for an input field UI.

    N = size(ui,2);
    f = (-(N/2)*df:df:(N/2-1)*df) + fo;
    tf = (fbw)/2/pi./((f-fc).^2+(fbw/2)^2);
    tf = tf/max(tf(:));
end
