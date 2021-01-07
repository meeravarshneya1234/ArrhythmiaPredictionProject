function APD90 = find_APD90(t,V)
%--------------------------------------------------------------------------
                        %% -- find_APD90.m -- %%
% Description: computes action potential duration at 90% repolarization 

% Inputs:
% --> t - [double array] time array   
% --> V - [double array] voltage array 

% Outputs: 
% --> APD90 - [double] computes action potential duration at 90%
% repolarization
% -------------------------------------------------------------------------
%%

t = t - t(1);
Vderiv = diff(V)./diff(t) ;
[dVdtmax,dexmax] = max(Vderiv) ;
tinit = t(dexmax(1));  %Time of maximum dV/dt, consider this beginning of action potential
vrest = min(V(1:dexmax));
[peakV,peakdex] = max(V) ;
tpeak = t(peakdex) ;
V90_exact = (0.1*(peakV - vrest) + vrest) ;

tspan = linspace(t(1),t(end),556);
vtemp = interp1(t,V,tspan);
V90dex = find(tspan > tpeak & vtemp < V90_exact);

if ~isempty(V90dex)
    V90_time = tspan(V90dex(1));
    APD90 = V90_time - tinit;
else
    APD90 = NaN;
end

