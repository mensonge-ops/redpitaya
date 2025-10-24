function gain = gain_saturated2(Pin,gssdB,PsatdBm)
%GAIN_SATURATED2 Calculate the saturated gain coefficient of the amplifier.

    gss = 10^(gssdB/10);
    Psat = 10^((PsatdBm-30)/10);
    gain = gss/(1+Pin/Psat);
end
