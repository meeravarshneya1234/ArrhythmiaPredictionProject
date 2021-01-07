function threshold = Find_Threshold(settings,plot_flag)

    if ~exist('plot_flag','var')
        plot_flag = false;
    end

    pert = settings_blockcurrents;
    pert.GNa = 0;

    threshold = 0 ;
    anyap = 0 ;
    Isub = 0 ;
    stim_amp = 10 ;
    dex = 1 ; %% just for now, keep all aps to be sure algorithm working

    ts = [];
    Vs = [];
    while(threshold == 0)
        settings.stim_amp = stim_amp;
        X = runSim(settings,pert); % run simulation
        ts{end+1} = X.times;
        Vs{end+1} = X.states(:,1);
        
        [~,dex_max] = max(X.states(:,1)) ;

        if max(X.states(:,1)) > 20  && X.times(dex_max) < 160
            Isup = stim_amp ;
            anyap = 1 ;
            stim_amp = 0.5*(Isub + Isup) ;
            if ((Isup-Isub) < 0.5)
                threshold = Isup ;
            end
        else
            Isub = stim_amp ;
            if (anyap)
                stim_amp = 0.5*(Isub + Isup) ;
            else
                stim_amp = 1.4*Isub ;
            end
        end % if/else
        dex = dex + 1 ;
    end % while loop to determine threshold
    
    if plot_flag
        colors = jet(dex);
        dexs = num2cell(1:dex-1);
        figure
        hold on
        cellfun(@(x,y,dex) plot(x,y,'linewidth',2,'color',colors(dex,:)),ts,Vs,dexs)
        plot(ts{end},Vs{end},'k--','linewidth',3)
        ylabel('Voltage (mV)')
        xlabel('time(ms)')
        set(gca,'FontSize',18,'FontWeight','bold','FontName','Calibri','box','off')
        title(['Threshold = ' num2str(threshold)]);
    end

end