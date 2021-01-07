function [p,c] = model_parameters(celltype)
p.celltype = celltype;
p.Inject = 0;
p.Nao=140.0;
p.Cao=1.8;
p.Ko=5.4;

%physical constants
p.R=8314.0;
p.T=310.0;
p.F=96485.0;
p.Cm=1.0; %uF

%cell geometry
p.L=0.01;
p.rad=0.0011;
p.vcell=1000*3.14*p.rad*p.rad*p.L;
p.Ageo=2*3.14*p.rad*p.rad+2*3.14*p.rad*p.L;
p.Acap=2*p.Ageo;
p.vmyo=0.68*p.vcell;
p.vnsr=0.0552*p.vcell;
p.vjsr=0.0048*p.vcell;
p.vss=0.02*p.vcell;

%jsr constants
p.bt=4.75;
p.a_rel=0.5*p.bt;

% channel conductances 
c.G.GNa=75;
c.G.GNaL=0.0075;
c.G.Gto=0.02;
c.G.GKr_=0.046;
c.G.GKs_=0.0034;
c.G.GK1=0.1908;
c.G.Gncx=0.0008;
c.G.GKb=0.003;
c.G.GpCa=0.0005;
c.G.PCa_=0.0001;
c.G.Pnak=30;
c.G.PNab=3.75e-10;
c.G.PCab=2.5e-8;
c.G.SERCA_total = 1 ;
c.G.RyR_total = 1 ;
c.G.Leak_total = 1;
c.G.Trans_total = 1;

if  strcmp(celltype,'epi')==1
    c.GNaL=c.G.GNaL*0.6;
    c.Gto=c.G.Gto*4.0;
    c.GKr_=c.G.GKr_*1.3;
    c.GKs_=c.G.GKs_*1.4;
    c.GK1=c.G.GK1*1.2;
    c.Gncx=c.G.Gncx*1.1;
    c.GKb=c.G.GKb*0.6;
    c.PCa_=c.G.PCa_*1.2;
    c.Pnak=c.G.Pnak*0.9;    
elseif  strcmp(celltype,'mid')==1
    c.G.Gto=c.G.Gto*4.0;
    c.G.GKr_=c.G.GKr_*0.8;
    c.G.GK1=c.G.GK1*1.3;
    c.G.Gncx=c.G.Gncx*1.4;
    c.G.PCa_=c.G.PCa_*2.5;
    c.G.Pnak=c.G.Pnak*0.7;    
end

% channel gating 
c.p.pm = 1;
c.p.ph = 1;
c.p.pj = 1;
c.p.php = 1;
c.p.pjp = 1;
c.p.pmL = 1;
c.p.phL = 1;
c.p.phLp = 1;
c.p.pa = 1;
c.p.pif = 1;
c.p.pis = 1;
c.p.pap =1;
c.p.pipf = 1;
c.p.pips = 1;
c.p.pd = 1;
c.p.pf = 1;
c.p.pfcaf = 1;
c.p.pfcas = 1;
c.p.pjca = 1;
c.p.pfpf = 1;
c.p.pfcapf = 1;
c.p.pxrf = 1;
c.p.pxrs = 1;
c.p.pxs1 = 1;
c.p.pxs2 = 1;
c.p.pxk1 = 1;

% channel voltage dependencies  
c.V.Vm = 0;
c.V.Vh = 0;
c.V.Vj = 0;
c.V.Vhp = 0;
c.V.Vjp = 0;
c.V.VmL = 0;
c.V.VhL = 0;
c.V.VhLp = 0;
c.V.Va = 0;
c.V.Vi = 0;
c.V.Vap = 0;
c.V.Vip= 0;
c.V.Vd= 0;
c.V.Vf= 0;
c.V.Vfca= 0;
c.V.Vjca= 0;
c.V.Vfp= 0;
c.V.Vfcap= 0;
c.V.Vxr= 0;
c.V.Vxs1= 0;
c.V.Vxs2= 0;
%c.V.Vxk1= 0;
c.V.Vncx= 0;
c.V.Vnak= 0;

end
