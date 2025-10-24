function uo = filter_gauss(ui,f3dB,fc,n,fo,df)
%FILTER_GAUSS Apply an N-th order Gaussian spectral filter to a field.
%   UO = FILTER_GAUSS(UI,F3DB,FC,N,FO,DF) filters the field UI using a
%   Gaussian transfer function with 3 dB bandwidth F3DB centred at FC.

    Ui = fft(ui);
    N = size(Ui,2);
    f = fftshift((-(N/2)*df:df:(N/2-1)*df) + fo);
    Tf = exp(-log(sqrt(2))*(2/f3dB*(f-fc)).^(2*n));
    uo = ifft(Ui.*Tf);
end
