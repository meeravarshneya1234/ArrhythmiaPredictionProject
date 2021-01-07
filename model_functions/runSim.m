function datatable = runSim(settings,pert)
%% 1--- Load Model Parameters 
[p,c] = model_parameters(settings.celltype);

%% 2--- Load Model Initial Conditions of State Variables 
if ~isfield(settings,'ICs')
    y0 = ICs(settings.steady_state,settings.PCL);
    y0s = repmat(y0,settings.variations,1);
else 
    y0s = settings.ICs;
end

%% 3--- Define Simulation Protocol 
stim_starts = settings.stim_delay + settings.PCL*(0:settings.nBeats-1)  ;
stim_ends = stim_starts + settings.stim_dur ;

% Create intervals for each beat 
simints = 3*settings.nBeats ;
for i=1:settings.nBeats
    intervals(3*i-2,:) = [settings.PCL*(i-1),stim_starts(i)] ; %beginning 
    intervals(3*i-1,:) = [stim_starts(i),stim_ends(i)] ; %stimulus 
    intervals(3*i,:) = [stim_ends(i),settings.PCL*i] ; % stimulus ends 
end
tend = settings.nBeats*settings.PCL ;              % end of simulation, ms
intervals(end,:) = [stim_ends(end),tend] ;

% Determine when to apply stim_amp or 0 amp  
Istim = zeros(simints,1) ;
stimindices = 3*(1:settings.nBeats) - 1 ; % apply stimulus on second part of intervals
Istim(stimindices) = -settings.stim_amp ; 

%% 4--- Population Variables 
F_G = fieldnames(c.G);
F_p = fieldnames(c.p);
F_V = fieldnames(c.V);

if ~isfield(settings,'scalings')
    S1 = exp(settings.sigmaG*randn(length(F_G),settings.variations))' ; % same parameter variation for each pop
    S2 = exp(settings.sigmap*randn(length(F_p),settings.variations))' ; % same parameter variation for each pop
    S3 = (settings.sigmaV*randn(length(F_V),settings.variations))' ; % same parameter variation for each pop
    S = [S1 S2 S3];
else
    S = settings.scalings;
end

%% 5--- Define Perturbation Protocol 
[p,c]= perturbations(c,p,pert);
baselineparameters = c;

%% 6--- Run Loop 
for ii=1:settings.variations
    scaling = S(ii,:);
    c = scaling_factors(scaling,baselineparameters,F_G,F_p,F_V);
    statevar_i = y0s(ii,:);
    options = odeset('RelTol',1e-3,'AbsTol',1e-6);
    
    % stimulate cell
    t = 0 ;
    statevars = statevar_i ;
    for i=1:simints
        [post,posstatevars] = ode15s(@dydt_Ohara,intervals(i,:),statevar_i,options,Istim(i),p,c) ;
        t = [t;post(2:end)] ;
        statevars = [statevars;posstatevars(2:end,:)] ;
        statevar_i = posstatevars(end,:) ;
    end % for
    % Only save the number of beats specified in numbertokeep
    start = find( t == intervals(simints-3*settings.numbertokeep+1,1));
    t_final = t(start:end);
    statevars_final = statevars(start:end,:);
    
    datatable(ii).times =  t_final - min(t_final) ;
    datatable(ii).states = statevars_final;
    datatable(ii).scalings = scaling;
    datatable(ii).currents = dydt_Ohara(t,num2cell(statevars_final,1),0,p,c,0);
end

%% Nested Functions 
function [p,c]= perturbations(c,p,pert)
    c.G.GNa  = c.G.GNa  * pert.GNa;
    c.G.GNaL = c.G.GNaL * pert.GNaL;
    c.G.Gto  = c.G.Gto  * pert.Gto;
    c.G.GKr_ = c.G.GKr_ * pert.GKr;
    c.G.GKs_ = c.G.GKs_ * pert.GKs;
    c.G.GK1  = c.G.GK1  * pert.GK1;
    c.G.Gncx = c.G.Gncx * pert.GNCX;
    c.G.GKb  = c.G.GKb  * pert.GKb;
    c.G.GpCa = c.G.GpCa * pert.GpCa;
    c.G.PCa_ = c.G.PCa_ * pert.PCa;
    c.G.Pnak = c.G.Pnak * pert.NaK;
    c.G.PNab = c.G.PNab * pert.PNab;
    c.G.PCab = c.G.PCab * pert.PCab;
    c.G.SERCA_total = c.G.SERCA_total * pert.SERCA ;
    c.G.RyR_total   = c.G.RyR_total   * pert.RyR ;
    c.G.Leak_total  = c.G.Leak_total  * pert.Leak;
    c.G.Trans_total = c.G.Trans_total * pert.Trans;

    % protocols
    p.Ko = p.Ko * pert.Ko;
    p.Inject = pert.Inject;
    p.Cao = p.Cao * pert.Cao;
    p.Nao = p.Nao * pert.Nao;

function c = scaling_factors(scaling,baselineparameters,n_G,n_p,n_V)
    scalings.G = scaling(1:length(n_G));
    scalings.p = scaling(length(n_G)+1:length(n_p)+length(n_G));
    scalings.V = scaling(length(n_G)+length(n_p)+1:end);

    for iF = 1:length(n_G)
        aF = n_G{iF};
        c.G.(aF) = baselineparameters.G.(aF) * scalings.G(iF);
    end

    for iF = 1:length(n_p)
        aF = n_p{iF};
        c.p.(aF) = baselineparameters.p.(aF) * scalings.p(iF);
    end

    for iF = 1:length(n_V)
        aF = n_V{iF};
        c.V.(aF) = scalings.V(iF) + baselineparameters.V.(aF);
    end