function [Dn] = norma(D)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Function normalizes matrix over [0,1]
% Dn:   normalized vector over [0,1]
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if max(D) == min (D)
    Dn = zeros(length(D),1);
else
   Dn = (D - min(D))./(max(D)-min(D));
end;
