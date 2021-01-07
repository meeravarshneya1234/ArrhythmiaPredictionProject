function out = dydt_Ohara(t,statevar,Id,p,c,flag)
if ~exist('flag','var') || flag 
    flag = 1;
    statevar = num2cell(statevar);    
end 

[V, Nai, Nass, Ki, Kss, Cai, Cass, Cansr, Cajsr, m, hf, hs, j, hsp, jp, mL, hL, hLp, a, iF, ...
    iS, ap, iFp, iSp, d, ff, fs, fcaf, fcas, jca, nca, ffp, fcafp, xrf, xrs, xs1, xs2, ...
    xk1, Jrelnp, Jrelp, CaMKt] = deal(statevar{:}) ;

%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
%% Compute ionic currents
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%

%CaMK constants
KmCaMK=0.15;
aCaMK=0.05;
bCaMK=0.00068;
CaMKo=0.05;
KmCaM=0.0015;
%update CaMK
CaMKb=CaMKo.*(1.0-CaMKt)./(1.0+KmCaM./Cass);
CaMKa=CaMKb+CaMKt;

dCaMKt=aCaMK.*CaMKb.*(CaMKb+CaMKt)-bCaMK.*CaMKt;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%reversal potentials
ENa=(p.R*p.T/p.F).*log(p.Nao./Nai);
EK=(p.R*p.T/p.F).*log(p.Ko./Ki);
PKNa=0.01833;
EKs=(p.R*p.T/p.F).*log((p.Ko+PKNa*p.Nao)./(Ki+PKNa.*Nai));

%convenient shorthand calculations
vffrt=V*p.F*p.F/(p.R*p.T);
vfrt=V*p.F/(p.R*p.T);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate INa

% m gate activation of fast INa
mss=1.0/(1.0+exp((-((V+c.V.Vm)+39.57))/9.871));
tm=c.p.pm*(1.0/(6.765*exp((V+11.64)/34.77)+8.552*exp(-(V+77.42)/5.955)));
dm=(mss-m)./tm;

% h gate inactivation of fast INa

% % % % % % hss=1.0/(1+exp(((V+Vh)+82.90)/6.086));
% % % % % % thf=phf*(1.0/(1.432e-5*exp(-(V+1.196)/6.285)+6.149*exp((V+0.5096)/20.27)));
% % % % % % ths=phs*(1.0/(0.009794*exp(-(V+17.95)/28.05)+0.3343*exp((V+5.730)/56.66)));
% % % % % % Ahf=0.99;
% % % % % % Ahs=1.0-Ahf;
% % % % % % dhf=(hss-hf)/thf;
% % % % % % dhs=(hss-hs)/ths;
% % % % % % h=Ahf*hf+Ahs*hs;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% combine fast and slow component, assigin only 1 Vshift and only 1 p
hss=1.0/(1+exp(((V+c.V.Vh)+82.90)/6.086));
% use an uniform ph instead of phf or phs
thf=c.p.ph*(1.0/(1.432e-5*exp(-(V+1.196)/6.285)+6.149*exp((V+0.5096)/20.27)));
ths=c.p.ph*(1.0/(0.009794*exp(-(V+17.95)/28.05)+0.3343*exp((V+5.730)/56.66)));
Ahf=0.99;
Ahs=1.0-Ahf;
dhf=(hss-hf)./thf;
dhs=(hss-hs)./ths;
h=Ahf*hf+Ahs*hs;


% j gate recovery of inactivation for fast INa
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jss=hss;
jss = 1.0/(1+exp(((V+c.V.Vj)+82.90)/6.086)); % write the equation again to avoid masking of hss alternation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tj=c.p.pj*(2.038+1.0/(0.02136*exp(-(V+100.6)/8.281)+0.3052*exp((V+0.9941)/38.45)));
dj=(jss-j)/tj;

% CaMK modulated h and j gate
hssp=1.0/(1+exp(((V+c.V.Vhp)+89.1)/6.086));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% thsp=3.0*ths; % write the equation again to avoid masking of ths alternation
thsp=c.p.php*(3.0*(1.0/(0.009794*exp(-(V+17.95)/28.05)+0.3343*exp((V+5.730)/56.66))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dhsp=(hssp-hsp)./thsp;
hp=Ahf*hf+Ahs*hsp;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jpss = jss ; % write the equation again to avoid masking of jss alternation
jpss = 1.0/(1+exp(((V+c.V.Vjp)+82.90)/6.086)); 
% tjp=1.46*tj; % write the equation again to avoid masking of tj alternation
tjp=c.p.pjp*(1.46*(2.038+1.0/(0.02136*exp(-(V+100.6)/8.281)+0.3052*exp((V+0.9941)/38.45))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
djp=(jpss-jp)./tjp;
fINap=(1.0./(1.0+KmCaMK./CaMKa));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INa=c.G.GNa.*(V-ENa).*m.^3.0.*((1.0-fINap).*h.*j+fINap.*hp.*jp);



%% calculate INaL

%  mL gate activation of INaL
mLss=1.0/(1.0+exp((-((V+c.V.VmL)+42.85))/5.264));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tmL=tm; % write the equation again to avoid masking of tm alternation
tmL=c.p.pmL*(1.0/(6.765*exp((V+11.64)/34.77)+8.552*exp(-(V+77.42)/5.955)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dmL=(mLss-mL)./tmL;

% hL gate inactivation of INaL
hLss=1.0/(1.0+exp(((V+c.V.VhL)+87.61)/7.488));
thL=c.p.phL*200.0; % originally thL = 200.0
% HF remodeling should be kept constant in the same model
dhL=(hLss-hL)./thL;

% CaMK modulated hL gate
hLssp=1.0/(1.0+exp(((V+c.V.VhLp)+93.81)/7.488));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% thLp=3.0*thL; % write the equation again to avoid masking of thL alternation
thLp=c.p.phLp*(3.0*200.0); % HF remodeling should be kept constant
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dhLp=(hLssp-hLp)./thLp;

fINaLp=(1.0./(1.0+KmCaMK./CaMKa));
INaL=c.G.GNaL.*(V-ENa).*mL.*((1.0-fINaLp).*hL+fINaLp.*hLp);

%% calculate Ito

% a gate activation of Ito
ass=1.0/(1.0+exp((-((V+c.V.Va)-14.34))/14.82));
ta=c.p.pa.*(1.0515./(1.0/(1.2089*(1.0+exp(-(V-18.4099)./29.3814)))+3.5/(1.0+exp((V+100.0)/29.3814))));
da=(ass-a)./ta;

% i gate inactivation of Ito
iss=1.0/(1.0+exp(((V+c.V.Vi)+43.94)/5.711));
if strcmp(p.celltype,'epi')==1
    delta_epi=1.0-(0.95/(1.0+exp((V+70.0)/5.0)));
else
    delta_epi=1.0;
end
tiF=c.p.pif*(4.562+1/(0.3933*exp((-(V+100.0))/100.0)+0.08004*exp((V+50.0)/16.59)));
tiS=c.p.pis*(23.62+1/(0.001416*exp((-(V+96.52))/59.05)+1.780e-8*exp((V+114.1)/8.079)));
tiF=tiF*delta_epi;
tiS=tiS*delta_epi;
AiF=1.0./(1.0+exp((V-213.6)./151.2));
AiS=1.0-AiF;
diF=(iss-iF)/tiF;
diS=(iss-iS)/tiS;
i=AiF.*iF+AiS.*iS;

% ap CaMK modulated activation of Ito
assp=1.0/(1.0+exp((-((V+c.V.Vap)-24.34))/14.82));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tap = ta ; % write the equation again to avoid masking of ta alternation
tap = c.p.pap.*(1.0515./(1.0/(1.2089*(1.0+exp(-(V-18.4099)./29.3814)))+3.5./(1.0+exp((V+100.0)/29.3814)))) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dap=(assp-ap)./tap;

% ip CaMK modulated inactivation of Ito
dti_develop=1.354+1.0e-4/(exp((V-167.4)/15.89)+exp(-(V-12.23)/0.2154));
dti_recover=1.0-0.5/(1.0+exp((V+70.0)/20.0));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% issp = iss ;    % write the equation again to avoid masking of iss/tiF/tiS alternation
% tiFp=dti_develop*dti_recover*tiF;
% tiSp=dti_develop*dti_recover*tiS;
issp = 1.0/(1.0+exp(((V+c.V.Vip)+43.94)/5.711)); 
tiFp=c.p.pipf.*(dti_develop.*dti_recover*(4.562+1./(0.3933*exp((-(V+100.0))./100.0)+0.08004*exp((V+50.0)/16.59)))*delta_epi);
tiSp=c.p.pips.*(dti_develop.*dti_recover*(23.62+1./(0.001416*exp((-(V+96.52))./59.05)+1.780e-8*exp((V+114.1)/8.079)))*delta_epi);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
diFp=(issp-iFp)./tiFp;
diSp=(issp-iSp)./tiSp;
ip=AiF.*iFp+AiS.*iSp;
fItop=(1.0./(1.0+KmCaMK./CaMKa));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Ito = c.G.Gto.*(V-EK).*((1.0-fItop).*a.*i+fItop.*ap.*ip);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate ICaL, ICaNa, ICaK

% d gate activation for ICaL
dss=1.0/(1.0+exp((-((V+c.V.Vd)+3.940))/4.230));
td=c.p.pd*(0.6+1.0/(exp(-0.05*(V+6.0))+exp(0.09*(V+14.0))));
dd=(dss-d)./td;

% f gate voltage dependent inactivation of ICaL

% % % % % fss=1.0/(1.0+exp(((V+Vf)+19.58)/3.696));
% % % % % tff=pff*(7.0+1.0/(0.0045*exp(-(V+20.0)/10.0)+0.0045*exp((V+20.0)/10.0)));
% % % % % tfs=pfs*(1000.0+1.0/(0.000035*exp(-(V+5.0)/4.0)+0.000035*exp((V+5.0)/6.0)));
% % % % % Aff=0.6;
% % % % % Afs=1.0-Aff;
% % % % % dff=(fss-ff)/tff;
% % % % % dfs=(fss-fs)/tfs;
% % % % % f=Aff*ff+Afs*fs;
% combine time constants for fast and slow gating
% use an uniform pf instead of pff and pfs
fss=1.0/(1.0+exp(((V+c.V.Vf)+19.58)/3.696));
tff=c.p.pf*(7.0+1.0/(0.0045*exp(-(V+20.0)/10.0)+0.0045*exp((V+20.0)/10.0)));
tfs=c.p.pf*(1000.0+1.0/(0.000035*exp(-(V+5.0)/4.0)+0.000035*exp((V+5.0)/6.0)));
Aff=0.6;
Afs=1.0-Aff;
dff=(fss-ff)./tff;
dfs=(fss-fs)./tfs;
f=Aff*ff+Afs*fs;

% fca gate Ca dependent inactivation of ICaL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fcass=fss; % write the equation again to avoid masking of fss alternation
fcass=1.0./(1.0+exp(((V+c.V.Vfca)+19.58)/3.696));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tfcaf=c.p.pfcaf*(7.0+1.0/(0.04*exp(-(V-4.0)/7.0)+0.04*exp((V-4.0)/7.0)));
tfcas=c.p.pfcas*(100.0+1.0/(0.00012*exp(-V/3.0)+0.00012*exp(V/7.0)));
Afcaf=0.3+0.6./(1.0+exp((V-10.0)/10.0));
Afcas=1.0-Afcaf;
dfcaf=(fcass-fcaf)./tfcaf;
dfcas=(fcass-fcas)./tfcas;

fca=Afcaf.*fcaf+Afcas.*fcas;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% jca gate recovery from Ca dependent inactivation of ICaL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% jcass = fcass ; % write the equation again to avoid masking of fcass alternation
jcass = 1.0/(1.0+exp(((V+c.V.Vjca)+19.58)/3.696)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
tjca=c.p.pjca*75.0;
djca=(jcass-jca)./tjca;

% fp (fast AND slow fp) gate, CaMK modulated voltage dependent inactivation of ICaL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fpss = fss ;
% tffp=2.5*tff; % write the equation again to avoid masking of fss/tff alternation
fpss = 1.0/(1.0+exp(((V+c.V.Vfp)+19.58)/3.696)) ;
tffp=c.p.pfpf*(2.5*(7.0+1.0/(0.0045*exp(-(V+20.0)/10.0)+0.0045*exp((V+20.0)/10.0))));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dffp=(fpss-ffp)./tffp;
fp=Aff.*ffp+Afs.*fs;

% fcap (fast AND slow fcap) gate, CaMK modulated Ca dependent inactivation of ICaL
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% tfcafp=2.5*tfcaf;
% fcapss = fcass ; % write the equation again to avoid masking of tfcaf/fcass alternation
tfcafp=c.p.pfcapf*(2.5*(7.0+1.0/(0.04*exp(-(V-4.0)/7.0)+0.04*exp((V-4.0)/7.0))));
fcapss = 1.0/(1.0+exp(((V+c.V.Vfcap)+19.58)/3.696)) ;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
dfcafp=(fcapss-fcafp)./tfcafp;
fcap=Afcaf.*fcafp+Afcas.*fcas;

Kmn=0.002;
k2n=1000.0;
km2n=jca*1.0;
anca=1.0./(k2n/km2n+(1.0+Kmn/Cass).^4.0);
dnca=anca.*k2n-nca.*km2n;
PhiCaL=4.0*vffrt.*(Cass.*exp(2.0*vfrt)-0.341*p.Cao)./(exp(2.0*vfrt)-1.0);
PhiCaNa=1.0*vffrt.*(0.75*Nass.*exp(1.0*vfrt)-0.75*p.Nao)./(exp(1.0*vfrt)-1.0);
PhiCaK=1.0*vffrt.*(0.75*Kss.*exp(1.0*vfrt)-0.75*p.Ko)./(exp(1.0*vfrt)-1.0);
zca=2.0;
PCap=1.1*c.G.PCa_;
PCaNa=0.00125*c.G.PCa_;
PCaK=3.574e-4*c.G.PCa_;
PCaNap=0.00125*PCap;
PCaKp=3.574e-4*PCap;
fICaLp=(1.0./(1.0+KmCaMK./CaMKa));
ICaL=(1.0-fICaLp).*c.G.PCa_.*PhiCaL.*d.*(f.*(1.0-nca)+jca.*fca.*nca)+fICaLp*PCap.*PhiCaL.*d.*(fp.*(1.0-nca)+jca.*fcap.*nca);
ICaNa=(1.0-fICaLp).*PCaNa.*PhiCaNa.*d.*(f.*(1.0-nca)+jca.*fca.*nca)+fICaLp*PCaNap.*PhiCaNa.*d.*(fp.*(1.0-nca)+jca.*fcap.*nca);
ICaK=(1.0-fICaLp).*PCaK.*PhiCaK.*d.*(f.*(1.0-nca)+jca.*fca.*nca)+fICaLp*PCaKp.*PhiCaK.*d.*(fp.*(1.0-nca)+jca.*fcap.*nca);

%% calculate IKr

% xr (fast AND slow) gate, activation/deactivation of IKr
xrss=1.0/(1.0+exp((-((V+c.V.Vxr)+8.337))/6.789));
txrf=c.p.pxrf*(12.98+1.0/(0.3652*exp((V-31.66)/3.869)+4.123e-5*exp((-(V-47.78))/20.38)));
txrs=c.p.pxrs*(1.865+1.0/(0.06629*exp((V-34.70)/7.355)+1.128e-5*exp((-(V-29.74))/25.94)));
Axrf=1.0./(1.0+exp((V+54.81)/38.21));
Axrs=1.0-Axrf;
dxrf=(xrss-xrf)./txrf;
dxrs=(xrss-xrs)./txrs;
xr=Axrf.*xrf+Axrs.*xrs;
rkr=1.0./(1.0+exp((V+55.0)/75.0))*1.0./(1.0+exp((V-10.0)/30.0));
IKr=c.G.GKr_*sqrt(p.Ko/5.4).*xr.*rkr.*(V-EK);

%% calculate IKs IK1

% xs1 gate activation of IKs
xs1ss=1.0/(1.0+exp((-((V+c.V.Vxs1)+11.60))/8.932));
txs1=c.p.pxs1*(817.3+1.0/(2.326e-4*exp((V+48.28)/17.80)+0.001292*exp((-(V+210.0))/230.0)));
dxs1=(xs1ss-xs1)./txs1;

% xs2 gate deactivation of IKs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% xs2ss=xs1ss;  % write the equation again to avoid masking of xs1ss alternation
xs2ss=1.0/(1.0+exp((-((V+c.V.Vxs2)+11.60))/8.932));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
txs2=c.p.pxs2*(1.0/(0.01*exp((V-50.0)/20.0)+0.0193*exp((-(V+66.54))/31.0)));
dxs2=(xs2ss-xs2)./txs2;
KsCa=1.0+0.6./(1.0+(3.8e-5./Cai).^1.4);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IKs = c.G.GKs_.*KsCa.*xs1.*xs2.*(V-EKs);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% xk1 gate inactivation of IK1
xk1ss=1.0/(1.0+exp(-((V)+2.5538*p.Ko+144.59)/(1.5692*p.Ko+3.8115)));
txk1=c.p.pxk1*(122.2/(exp((-(V+127.2))/20.36)+exp((V+236.8)/69.33)));
dxk1=(xk1ss-xk1)./txk1;
rk1=1.0./(1.0+exp((V+105.8-2.6*p.Ko)./9.493));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
IK1 = c.G.GK1.*sqrt(p.Ko).*rk1.*xk1.*(V-EK);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate INaCa_i
kna1=15.0;
kna2=5.0;
kna3=88.12;
kasymm=12.5;
wna=6.0e4;
wca=6.0e4;
wnaca=5.0e3;
kcaon=1.5e6;
kcaoff=5.0e3;
qna=0.5224;
qca=0.1670;
hca=exp((qca*(V+c.V.Vncx)*p.F)/(p.R*p.T));
hna=exp((qna*(V+c.V.Vncx)*p.F)/(p.R*p.T));
h1=1+Nai./kna3.*(1+hna);
h2=(Nai.*hna)./(kna3.*h1);
h3=1.0./h1;
h4=1.0+Nai./kna1.*(1+Nai./kna2);
h5=Nai.*Nai./(h4.*kna1.*kna2);
h6=1.0./h4;
h7=1.0+p.Nao./kna3*(1.0+1.0./hna);
h8=p.Nao./(kna3.*hna.*h7);
h9=1.0./h7;
h10=kasymm+1.0+p.Nao./kna1*(1.0+p.Nao/kna2);
h11=p.Nao*p.Nao./(h10.*kna1.*kna2);
h12=1.0./h10;
k1=h12.*p.Cao.*kcaon;
k2=kcaoff;
k3p=h9*wca;
k3pp=h8*wnaca;
k3=k3p+k3pp;
k4p=h3.*wca./hca;
k4pp=h2*wnaca;
k4=k4p+k4pp;
k5=kcaoff;
k6=h6.*Cai.*kcaon;
k7=h5.*h2.*wna;
k8=h8.*h11.*wna;
x1=k2.*k4.*(k7+k6)+k5.*k7.*(k2+k3);
x2=k1.*k7.*(k4+k5)+k4.*k6.*(k1+k8);
x3=k1.*k3.*(k7+k6)+k8.*k6.*(k2+k3);
x4=k2.*k8.*(k4+k5)+k3.*k5.*(k1+k8);
E1=x1./(x1+x2+x3+x4);
E2=x2./(x1+x2+x3+x4);
E3=x3./(x1+x2+x3+x4);
E4=x4./(x1+x2+x3+x4);
KmCaAct=150.0e-6;
allo=1.0./(1.0+(KmCaAct./Cai).^2.0);
zna=1.0;
JncxNa=3.0.*(E4.*k7-E1.*k8)+E3.*k4pp-E2.*k3pp;
JncxCa=E2.*k2-E1.*k1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INaCa_i = 0.8.*c.G.Gncx.*allo.*(zna.*JncxNa+zca.*JncxCa);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate INaCa_ss
h1=1+Nass./kna3.*(1+hna);
h2=(Nass.*hna)./(kna3.*h1);
h3=1.0./h1;
h4=1.0+Nass./kna1.*(1+Nass./kna2);
h5=Nass.*Nass./(h4.*kna1.*kna2);
h6=1.0./h4;
h7=1.0+p.Nao./kna3*(1.0+1.0./hna);
h8=p.Nao./(kna3.*hna.*h7);
h9=1.0./h7;
h10=kasymm+1.0+p.Nao./kna1.*(1+p.Nao./kna2);
h11=p.Nao*p.Nao./(h10*kna1*kna2);
h12=1.0./h10;
k1=h12*p.Cao*kcaon;
k2=kcaoff;
k3p=h9.*wca;
k3pp=h8.*wnaca;
k3=k3p+k3pp;
k4p=h3.*wca./hca;
k4pp=h2.*wnaca;
k4=k4p+k4pp;
k5=kcaoff;
k6=h6.*Cass*kcaon;
k7=h5.*h2.*wna;
k8=h8.*h11*wna;
x1=k2.*k4.*(k7+k6)+k5.*k7.*(k2+k3);
x2=k1.*k7.*(k4+k5)+k4.*k6.*(k1+k8);
x3=k1.*k3.*(k7+k6)+k8.*k6.*(k2+k3);
x4=k2.*k8.*(k4+k5)+k3.*k5.*(k1+k8);
E1=x1./(x1+x2+x3+x4);
E2=x2./(x1+x2+x3+x4);
E3=x3./(x1+x2+x3+x4);
E4=x4./(x1+x2+x3+x4);
KmCaAct=150.0e-6;
allo=1.0./(1.0+(KmCaAct./Cass).^2.0);
JncxNa=3.0.*(E4.*k7-E1.*k8)+E3.*k4pp-E2.*k3pp;
JncxCa=E2.*k2-E1.*k1;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
INaCa_ss = 0.2.*c.G.Gncx.*allo.*(zna.*JncxNa+zca.*JncxCa);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate INaK
k1p=949.5;
k1m=182.4;
k2p=687.2;
k2m=39.4;
k3p=1899.0;
k3m=79300.0;
k4p=639.0;
k4m=40.0;
Knai0=9.073;
Knao0=27.78;
delta=-0.1550;
Knai=Knai0*exp((delta*(V+c.V.Vnak)*p.F)./(3.0*p.R*p.T));
Knao=Knao0*exp(((1.0-delta)*(V+c.V.Vnak)*p.F)./(3.0*p.R*p.T));
Kki=0.5;
Kko=0.3582;
MgADP=0.05;
MgATP=9.8;
Kmgatp=1.698e-7;
H=1.0e-7;
eP=4.2;
Khp=1.698e-7;
Knap=224.0;
Kxkur=292.0;

P=eP./(1.0+H./Khp+Nai./Knap+Ki./Kxkur);
a1=(k1p*(Nai./Knai).^3.0)./((1.0+Nai./Knai).^3.0+(1.0+Ki./Kki).^2.0-1.0);
b1=k1m*MgADP;
a2=k2p;
b2=(k2m*(p.Nao./Knao).^3.0)./((1.0+p.Nao./Knao).^3.0+(1.0+p.Ko./Kko).^2.0-1.0);
a3=(k3p.*(p.Ko./Kko).^2.0)./((1.0+p.Nao./Knao).^3.0+(1.0+p.Ko./Kko).^2.0-1.0);
b3=(k3m*P*H)./(1.0+MgATP/Kmgatp);
a4=(k4p*MgATP/Kmgatp)./(1.0+MgATP/Kmgatp);
b4=(k4m*(Ki./Kki).^2.0)./((1.0+Nai./Knai).^3.0+(1.0+Ki./Kki).^2.0-1.0);
x1=a4.*a1.*a2+b2.*b4.*b3+a2.*b4.*b3+b3.*a1.*a2;
x2=b2.*b1.*b4+a1.*a2.*a3+a3.*b1.*b4+a2.*a3.*b4;
x3=a2.*a3.*a4+b3.*b2.*b1+b2.*b1.*a4+a3.*a4.*b1;
x4=b4.*b3.*b2+a3.*a4.*a1+b2.*a4.*a1+b3.*b2.*a1;
E1=x1./(x1+x2+x3+x4);
E2=x2./(x1+x2+x3+x4);
E3=x3./(x1+x2+x3+x4);
E4=x4./(x1+x2+x3+x4);
zk=1.0;
JnakNa=3.0.*(E1.*a3-E2.*b3);
JnakK=2.0.*(E4.*b1-E3.*a1);
INaK=c.G.Pnak.*(zna.*JnakNa+zk.*JnakK);

%% calculate IKb
xkb=1.0./(1.0+exp(-(V-14.48)./18.34));
IKb=c.G.GKb.*xkb.*(V-EK);

%% calculate INab
INab=c.G.PNab.*vffrt.*(Nai.*exp(vfrt)-p.Nao)./(exp(vfrt)-1.0);

%% calculate ICab
ICab=c.G.PCab.*4.0.*vffrt.*(Cai.*exp(2.0.*vfrt)-0.341.*p.Cao)./(exp(2.0.*vfrt)-1.0);

%% calculate IpCa
IpCa=c.G.GpCa.*Cai./(0.0005+Cai);

%% updtate the membrane voltage
% used for injecting current 
if V >= -60 
    dv=-(1/p.Cm)*(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+Id-p.Inject);
else
    dv=-(1/p.Cm)*(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+Id);
end
%dv=-(1/p.Cm)*(INa+INaL+Ito+ICaL+ICaNa+ICaK+IKr+IKs+IK1+INaCa_i+INaCa_ss+INaK+INab+IKb+IpCa+ICab+Id);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate diffusion fluxes
JdiffNa=(Nass-Nai)/2.0;
JdiffK=(Kss-Ki)/2.0;
Jdiff=(Cass-Cai)/0.2;

%% calculate ryanodione receptor calcium induced calcium release from the jsr
bt=4.75;
a_rel=0.5*bt;
Jrel_inf=a_rel.*(-ICaL)./(1.0+(1.5/Cajsr).^8.0);
if strcmp(p.celltype,'mid')==1
    Jrel_inf=Jrel_inf*1.7;
end
tau_rel=bt./(1.0+0.0123/Cajsr);

if tau_rel<0.001
   tau_rel=0.001; 
end

dJrelnp=(Jrel_inf-Jrelnp)./tau_rel;
btp=1.25*bt;
a_relp=0.5*btp;
Jrel_infp=a_relp.*(-ICaL)./(1.0+(1.5/Cajsr).^8.0);
if strcmp(p.celltype,'mid')==1
    Jrel_infp=Jrel_infp*1.7;
end
tau_relp=btp./(1.0+0.0123./Cajsr);

if tau_relp<0.001
   tau_relp=0.001; 
end

dJrelp=(Jrel_infp-Jrelp)./tau_relp;
fJrelp=(1.0./(1.0+KmCaMK./CaMKa));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Jrel= c.G.RyR_total.*((1.0-fJrelp).*Jrelnp+fJrelp.*Jrelp );
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calculate serca pump, ca uptake flux
Jupnp=0.004375.*Cai./(Cai+0.00092);
Jupp=2.75*0.004375.*Cai./(Cai+0.00092-0.00017);
if strcmp(p.celltype,'epi')==1
    Jupnp=Jupnp*1.3;
    Jupp=Jupp*1.3;
end
fJupp=(1.0./(1.0+KmCaMK./CaMKa));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Jleak = c.G.Leak_total*0.0039375*Cansr/15.0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Jup=c.G.SERCA_total.*((1.0-fJupp).*Jupnp+fJupp.*Jupp) - Jleak;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% calculate tranlocation flux
ttr = 100.0 ;
Jtr=c.G.Trans_total.*(Cansr-Cajsr)./ttr;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%calcium buffer constants
cmdnmax=0.05;
if strcmp(p.celltype,'epi')==1
    cmdnmax=cmdnmax*1.3;
end
kmcmdn=0.00238;
trpnmax=0.07;
kmtrpn=0.0005;
BSRmax=0.047;
KmBSR=0.00087;
BSLmax=1.124;
KmBSL=0.0087;
csqnmax=10.0;
kmcsqn=0.8;

%update intracellular concentrations, using buffers for cai, cass, cajsr
dnai=-(INa+INaL+3.0*INaCa_i+3.0*INaK+INab)*p.Acap/(p.F*p.vmyo)+JdiffNa*p.vss/p.vmyo;
dnass=-(ICaNa+3.0*INaCa_ss)*p.Acap/(p.F*p.vss)-JdiffNa;

dki=-(Ito+IKr+IKs+IK1+IKb+Id-2.0*INaK)*p.Acap/(p.F*p.vmyo)+JdiffK*p.vss/p.vmyo;
dkss=-(ICaK)*p.Acap/(p.F*p.vss)-JdiffK;

Bcai=1.0./(1.0+cmdnmax*kmcmdn./(kmcmdn+Cai).^2.0+trpnmax*kmtrpn./(kmtrpn+Cai).^2.0);
dcai=Bcai.*(-(IpCa+ICab-2.0*INaCa_i).*p.Acap./(2.0*p.F*p.vmyo)-Jup*p.vnsr/p.vmyo+Jdiff*p.vss/p.vmyo);

Bcass=1.0./(1.0+BSRmax*KmBSR./(KmBSR+Cass).^2.0+BSLmax*KmBSL./(KmBSL+Cass).^2.0);
dcass=Bcass.*(-(ICaL-2.0*INaCa_ss).*p.Acap./(2.0*p.F*p.vss)+Jrel*p.vjsr/p.vss-Jdiff);

dcansr=Jup-Jtr*p.vjsr/p.vnsr;

Bcajsr=1.0./(1.0+csqnmax*kmcsqn./(kmcsqn+Cajsr).^2.0);
dcajsr=Bcajsr.*(Jtr-Jrel);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if flag
    out=[dv dnai dnass dki dkss dcai dcass dcansr dcajsr dm dhf dhs ...
        dj dhsp djp dmL dhL dhLp da diF diS dap diFp diSp dd dff dfs dfcaf ...
        dfcas djca dnca dffp dfcafp dxrf dxrs dxs1 dxs2 dxk1 dJrelnp dJrelp dCaMKt]';
else
    
    out = [INa INaL Ito ICaL IKr IKs IK1 INaCa_i INaCa_ss INaK IKb...
        INab ICab IpCa Jrel Jup Jtr];
end

