classdef popfuncs
    
    methods(Static)
%--------------------------------------------------------------------------
                      %% -- create_scale_vector.m -- %%
% Description: creates population variability scaling matrix.  

% Inputs:
% --> settings - [struct] simulation protocol settings    
% --> variations - [double array] number of members of the population 

% Outputs: 
% --> scaling_matrix - [double array] population variability scaling matrix
% -------------------------------------------------------------------------
        function scaling_matrix = create_scale_vector(settings,variations)                     
            [~,c] = model_parameters('endo');
            nG = length(fieldnames(c.G));
            np = length(fieldnames(c.p));
            nV = length(fieldnames(c.V));
            
            S1 = exp(settings.sigmaG*randn(nG,variations))' ; % same parameter variation for each pop
            S2 = exp(settings.sigmap*randn(np,variations))' ;
            S3 = (settings.sigmaV*randn(nV,variations))' ;
            
            scaling_matrix = [S1,S2,S3];
                
        end 
               
%--------------------------------------------------------------------------
                      %% -- clean_population.m -- %%
% Description: Determines if there are arrhythmias in the population,
% removes them, and reruns new cells 

% Inputs:
% --> settings - [struct] simulation protocol settings    
% --> Xin - [struct] population time and state variables  

% Outputs: 
% --> Xout - [struct] cleaned population 
% -------------------------------------------------------------------------
        function Xout = clean_population(Xin,settings)
            flag = 1; % run the while loop
            settings_rerun = settings;
            X2 = Xin; total_variations = settings.variations;

            while flag     
                if settings.remove_arrhythmias
                    removes_1 = popfuncs.find_arrhythmias(settings_rerun,X2); % remove cells with arrhythmias before trigger
                    X2(removes_1) = [];
                elseif settings.remove_experimental
                    removes_2 = popfuncs.calibrate_experimental(settings_rerun,X2); % remove cells that are not within experimental range
                    X2(removes_2) = [];
                end
                n_X2 = length(X2); % # of cells kept
                
                if length(X2) == total_variations || settings.reruncells == 0 
                    flag = 0;
                else
                    settings_rerun.variations = total_variations - n_X2; % number of cells to rerun
                    settings_rerun.scalings = popfuncs.create_scale_vector(settings_rerun,settings_rerun.variations);
                    disp(['Need to Rerun: ' num2str(settings_rerun.variations) ' cells.'])
                    
                    pert = settings_blockcurrents;
                    X = runSim(settings_rerun,pert); % run population simulation
                    X2 = [X2,X];
                end
            end
            Xout = X2;
        end
        
%--------------------------------------------------------------------------
                      %% -- find_arrhythmias.m -- %%
% Description: Determines if there are arrhythmias in a population

% Inputs:
% --> settings - [struct] simulation protocol settings    
% --> Xin - [struct] population time and state variables  

% Outputs: 
% --> remove_AP - [logical] logical array of which cells have arrhythmias  
% -------------------------------------------------------------------------
        function remove_AP = find_arrhythmias(settings,Xin)
            remove_AP = false(length(Xin),1);
            for ii = 1:length(Xin) %for each member of the population 
                Xt = Xin(ii).times;
                XStates = Xin(ii).states;
                [times,volts] = popfuncs.splitdata(Xt,XStates,settings); % saved the last 10 beats, separate each beat

                % check each beat for arrhythmic activity 
                for i = 1:settings.numbertokeep %for each beat 
                    t = times{i}; t = t - t(1);
                    V = volts{i}(:,1);
                    APDs(i) = find_APD90(t,V);
                    
                    if isnan(APDs(i)) || V(1) > -70 || max(V) < 15 %check if cell failed to repolarize
                        remove_AP(ii) = true;
                         
                    else %check AP for early afterdepolarizations 
                        [~,index] = find(t==settings.stim_delay);
                        time_roi = t(index:end);
                        Vm_roi = V(index:end);
                        dVm_roi = (Vm_roi(2:end)-Vm_roi(1:end-1))./(time_roi(2:end)-time_roi(1:end-1));
                        dVm_dVm = dVm_roi(1:end-1).*dVm_roi(2:end);
                        [~, idx_dVm] = max(dVm_roi);
                        dVm_0 = dVm_dVm(idx_dVm:end) < -1.0*10^-6;
                        dVm_t = time_roi(3:end);
                        tVm_0 = dVm_t(idx_dVm:end);
                        ts = tVm_0(dVm_0);
                        time_between_depols = diff(ts);
                        
                        if any(time_between_depols > 130)
                            remove_AP(ii) = true;  
                        end  
                    end  
                end
                if diff(APDs) > 5 %check for alternans  
                    remove_AP(ii) = true;
                end
            end
%             I = find(remove_AP);
%             Xout(I) = [];

        end
%--------------------------------------------------------------------------
                 %% -- calibrate_experimental.m -- %%
% Description:

% Inputs:
% --> settings - [struct] simulation protocol settings    
% --> Ti - [double array] time matrix 
% --> V - [double array] voltage matrix 

% Outputs: 
% --> times - [cell] time vector for each beat 
% --> volts - [cell] voltage vector for each beat  
%--------------------------------------------------------------------------
function remove_AP = calibrate_experimental(settings,Xin)
            remove_AP = false(length(Xin),1);
            for ii = 1:length(Xin) %for each member of the population
                Xt = Xin(ii).times;
                XState = Xin(ii).states;
                [times,volts,cais] = popfuncs.splitdata(Xt,XState,settings); % saved the last 10 beats, separate each beat

                t = times{end};
                V = volts{end};
                Cai = cais{end} * 1000000;
                
                [outputs,~] = calculate_features(V,Cai,t);
                
                outs = num2cell(outputs,1);
                [Vrest, Upstroke, Vpeak, APD20, APD40, APD50, APD90,...
                    TriAP,DCai, Capeak, CaD50, CaD90, TriCa, dCa] =deal(outs{:});
%                 APA = max(V) - min(V);
                
                I1 = APD20 > 50 && APD20 < 200;
                I2 = APD50 > 75 && APD50 < 400;
                I3 = APD90 > 100 && APD90 < 600;
%                 I4 = APA > 95 && APA < 130;
                I5 = CaD50 > 50 && CaD50 < 350;
                I6 = CaD90 > 200 && CaD90 < 700;
                I7 = DCai > 210 && DCai < 530;

                I = [I1,I2,I3,I5,I6,I7];

                if any(I == 0)
                    remove_AP(ii) = true;
                end
            end
        end
%--------------------------------------------------------------------------
                      %% -- splitdata.m -- %%
% Description: When multiple beats of a single cell are simulated, this
% function separates each beat into its own cell array. This is used
% mainly when settings.numbertokeep is greater than 1. 

% Inputs:
% --> settings - [struct] simulation protocol settings    
% --> Ti - [double array] time matrix 
% --> V - [double array] voltage matrix 

% Outputs: 
% --> times - [cell] time vector for each beat 
% --> volts - [cell] voltage vector for each beat 
% -------------------------------------------------------------------------
        function [times,volts,cais] = splitdata(Ti,States,settings)   
            numberofAPs = settings.numbertokeep;
            PCL = settings.PCL;
            intervals = find(~mod(Ti,PCL));
            times = {};
            volts ={};
            cais = {};
            for i = 1:numberofAPs
                times{end+1} = Ti(intervals(i):intervals(i+1));
                volts{end+1} = States(intervals(i):intervals(i+1),1);
                cais{end+1} = States(intervals(i):intervals(i+1),6);
            end
        end
    end 
end
