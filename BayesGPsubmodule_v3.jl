function B3fit(xt,yt)
    lt = length(xt)
    xfit,yfit = xt[lt-2:lt], yt[lt-2:lt]
    @. model(x, p) = p[1] + p[2]*exp(-x*p[3])
    p0 = [-35, 1.0, 1.0]
    ub = [yt[lt], Inf, Inf]
    fit = curve_fit(model, xfit, yfit, p0)
    pfit = coef(fit) 
    return pfit
end

function makeRm(xt,xp,lt,lp)
    Rtt=zeros(Float64,lt,lt);Rpt=zeros(Float64,lp,lt);Rpp=zeros(Float64,lp,lp)
    for i=1:lt
        for j=1:lt
            Rtt[i,j]=abs(log(xt[i])-log(xt[j]))
        end
    end
    for i=1:lp
        for j=1:lt
            Rpt[i,j]=abs(log(xp[i])-log(xt[j]))
        end
    end
    for i=1:lp
        for j=1:lp
            Rpp[i,j]=abs(log(xp[i])-log(xp[j]))
        end
    end
    return Rtt,Rpt,Rpp
end
function Resample(iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs,numN)
    w_der=[0.0 for i=1:numN]
    x=[ [] for i=1:numN]
    Pv = [ [iThetas[ith], yprds[ith], yds[ith], yd2s[ith],
            PHs[ith], logp0s[ith], llhs[ith], logposts[ith],
            ders[ith], mujs[ith], Sjs[ith]] for ith=1:numN]
    for i =1:numN
        tmp= ders[i][1] + ders[i][2]
        if tmp > 709.0
            w_der[i]= 1.e+30
        elseif tmp<-746.0
            w_der[i]=1.0e-100
        else
            w_der[i]=exp(tmp)
        end
    end
    StatsBase.alias_sample!(Pv,weights(w_der),x)
    iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs= [ [x[ith][jj] for ith=1:numN] for jj=1:11]
    return iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs
end                    

function readinput(inpname)
    tmp=split(inpname,"_")[(length(split(inpname,"_")))]
    if Kernel=="logMatern" || Kernel=="logRBF"
        Tsigma = [1.0 0.0;0.0 0.8]
        tTheta=[20.0,20]
    else    
        Tsigma = [0.1 0.0;0.0 0.1]
        tTheta=[1.0,10]
    end
    lines = open( inpname, "r" ) do inp; readlines(inp); end
    txall=[];hws=[]
    for line in lines 
        if string(line[1])==string("#") || string(line[1])==string("!");continue;end
        tl=split(line);tl=split(line,",")        
        hw = parse(Float64,tl[2])
        if (hw in hws)==false
            push!(hws,hw)
        end
        Nmax=parse(Float64,tl[1])
        if  Nmax in txall
            continue
        else
            push!(txall,Nmax)
        end
    end
    tyall=[ 0.0 for i=1:length(txall)]

    for line in lines 
        if string(line[1])==string("#") || string(line[1])==string("!");continue;end
        tl=split(line);tl=split(line,",")
        Nmax=parse(Float64,tl[1])
        for kk =1:length(txall)
            if txall[kk]== Nmax
                if multihw == true
                    push!(tyall[kk],parse(Float64,tl[3]))
                else
                    tyall[kk]=parse(Float64,tl[3])
                end
            end
        end
    end    
    xall=txall;yall=tyall
    ndata=length(xall)
    #### Data selection ####
    useind=collect(1:1:ndata)    
    ########################
    l1=length(useind);l2=ndata-l1
    oxt,oyt,xun,yun=make_xyt(xall,yall,useind,l1,l2,false)
    println("x(data) $oxt \ny(data) $oyt")
    xt=oxt;yt=oyt
    olt=length(xt)

    pfit=B3fit(xt,yt)
    println("B3fit $pfit")

    iThetas=[ [100*rand(),20.0*rand()] for i=1:numN]
    if Kernel=="logRBF" || Kernel=="logMatern"
        iThetas=[ [100*rand(),30.0*rand()] for i=1:numN]
    end
    if Kernel=="NSMatern" || Kernel=="Lin+Mat" || Kernel=="Lin*Mat"
        iThetas=[ [10*rand(),5.0*rand(),1.0*rand()] for i=1:numN]
    end

    lt=length(xt)
    Ev=0.0
    Mysigma = Diagonal( (1.e-12) * ones(Float64,lt,lt))

    #muy=mean(yt)
    muy=minimum(yt) ### (more efficient?)
    mstd=std(yt)

    xprd=Float64[];pNmax=[]
    unx=collect(xt[lt]+2:2.0:xpMax)
    for tmp in unx 
        if (tmp in xt)==false 
            push!(xprd,tmp)
        end
    end
    lp=length(xprd)
    Rms=makeRm(xt,xprd,lt,lp)
    return Tsigma,tTheta,xt,yt,xprd,xun,yun,oxt,oyt,iThetas,lt,lp,Mysigma,muy,mstd,pfit,Rms
end

function updateyprd(typrd::Array{Float64,1},cqY::Float64,Sjoint::Array{Float64,2})            
    try
        rand(MvNormal(typrd,cqY*Sjoint))
    catch
        errc = 0
        lp=length(typrd)
        while isposdef(cqY*Sjoint)==false
            ttmp=eigvals(Sjoint)
            minEV=ttmp[1]              
            Sjoint = Sjoint + Diagonal( 100.0*abs(minEV) * ones(Float64,lp,lp))
            errc += 1
            if errc > 100;println("NotPSD:Sjoint L516:@",Theta,ttmp);exit();end
        end
        rand(MvNormal(typrd,cqY*Sjoint))
    end
end

function updateyprd_n(typrd::Array{Float64,1},ln,mstd)
    typrd + 1.e-2 * randn(ln)
end

function calc_ratio(yprd,ytendm1,ytend,mstd,muy)
    yprd=yprd *mstd.+muy
    ytendm1 = ytendm1 *mstd.+muy
    ytend = ytend *mstd.+muy
    ln=length(yprd)-1
    b=[ 0.0 for i=1:ln]
    b[1] = (ytend-yprd[1])/(ytendm1 - ytend)
    b[2] = (yprd[1]-yprd[2])/(ytend - yprd[1])
    for i =3:ln
        b[i] = (yprd[i-1]-yprd[i])/(yprd[i-2]-yprd[i-1])
    end
    return b
end

function updateTheta(tTheta::Array{Float64,1},sigma_T::Array{Float64,2})
    c_Theta = exp.(rand(MvNormal(log.(tTheta),sigma_T)))
    while (c_Theta[1]>1.e+5 || c_Theta[1]<1.e-7 || c_Theta[2] > 50)
        c_Theta = exp.(rand(MvNormal(log.(tTheta),sigma_T)))
    end
    return c_Theta
end

function main(mstep,numN,sigRfac)
    Tsigma,tTheta,xt,yt,xprd,xun,yun,oxt,oyt,iThetas,lt,lp,Mysigma,muy,mstd,pfit,Rms=readinput(inpname)
    global iniThetas = deepcopy(iThetas)
    Rtt,Rpt,Rpp=Rms
    xd=[0.0]; xd2=[0.0]
    ld=length(xd);ld2=length(xd2)
    yprd=0*xprd.-1.e-2; yd=zeros(Float64,ld);  yd2=zeros(Float64,ld2)

    global R=(yt[lt-1]-yt[lt])/(yt[lt-2]-yt[lt-1])
    global sigR = sigRfac * R
    
    achitT=0;achitY=0
    cqT=qT;cqY=qY

    if paramean==true
        mu_yt  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xt[i])  for i =1:lt],lt).-muy)/mstd
        mu_yp  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i])    for i =1:lp],lp).-muy)/mstd
    else 
        mu_yt = reshape([ 0.0 for i =1:lt],lt); mu_yp = reshape([ 0.0 for i =1:lp],lp)
    end
    meanf = vcat(mu_yt,mu_yp)

    tmuj=zeros(Float64,lp+ld+ld2);tSj=ones(Float64,lp+ld+ld2,lp+ld+ld2)

    tPH= -1.e+300;tlogprior=-1.e+15;tllh=-1.e+15;tlogpost=-1.e+15;tder=-1.e+15;tder2=-1.e+15
    yprds= [ yprd for i=1:numN]; yds=[yd for i=1:numN];yd2s=[yd for i=1:numN]
    PHs = [ tPH for i=1:numN]; logp0s = [tlogprior for i=1:numN]
    llhs=[tllh for i=1:numN]; logposts=[tlogpost for i=1:numN]; ders=[ [tder,tder2] for i=1:numN]
    mujs=[ tmuj for i=1:numN];Sjs =[ tSj for i=1:numN]
  
    yt= (yt .-muy)/mstd
    ydel = (yt[lt-1]-yt[lt])   
    println("xp",xprd," xt ", xt," yt ",yt," muy ",muy,"mstd",mstd, "lp $lp ld $ld ld2 $ld2 ")
    Ktt0 = zeros(Float64,lt,lt);Kpp0 = zeros(Float64,lp,lp)
    Kpt0 = zeros(Float64,lp,lt);Ktp0 = zeros(Float64,lt,lp)
    if paramean==true;NDist=Normal(0,0.001);else; NDist=Normal(R,sigR);end

    tstep = 1
    for ith=1:numN 
        ## tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
        tyd=0*xd; tyd2=0*xd2
        Accept,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmuj,tSj = evalPH_t(Kernel,tstep,mstep,ith,xt,yt,lt,Mysigma,muy,mstd,xprd,yprd,lp,xd,yd,ld,xd2,yd2,ld2,iThetas[ith],Monotonic,Convex,"Sample",false,1.e+10,meanf,Rtt,Rpt,Rpp,Ktt0,Kpt0,Ktp0,Kpp0)
        if paramean==true
            pn=sign( 0.5-rand())
            c_yprd= [ mu_yp[i] + pn*0.005*mu_yp[i] for i=1:lp ]
        else
            rtmp = rand(NDist);c_yprd= [ yt[lt] - ydel * rtmp * (rtmp^(i)-1.0)/(rtmp-1.0) for i=1:lp]
        end
        tder,tder2 = eval_der(tstep,mstep,xt,yt,xprd,c_yprd,mstd,R,sigR)
        tPH = tlogprior+tllh+tlogpost+tder+tder2

        #iThetas[ith] = tTheta
        yprds[ith]= c_yprd; yds[ith]=tyd; yd2s[ith]=tyd2
        PHs[ith]=tPH; logp0s[ith]=tlogprior; llhs[ith]=tllh; logposts[ith]=tlogpost
        ders[ith]=[tder,tder2];mujs[ith]=tmuj;Sjs[ith]= tSj
    end
    println("yprds[1]:", yprds[1] *mstd .+ muy)

    sigma_T = cqT*Tsigma
    c_yd=0.0*xd;c_yd2=xd2*0.0   #### to be changed if derKer introduced         
    for tstep=2:mstep
        println("tstep ",tstep)
        #@threads for ith=1:numN 
#        @time @simd for ith=1:numN
        @time @inbounds @simd for ith=1:numN 
            #t0 =time()
            tTheta=iThetas[ith]
            tder,tder2=ders[ith]
            tPH,tlogprior,tllh,tlogpost = PHs[ith],logp0s[ith], llhs[ith],logposts[ith]
            tyd=yds[ith]; tyd2=yd2s[ith]
            prePH = tlogprior + tllh + tlogpost
            
            #t1 = time()

            c_Theta = updateTheta(tTheta,sigma_T)

            #t2 = time()
 
            tmp=evalPH_t(Kernel,tstep,mstep,ith,xt,yt,lt,
                         Mysigma,muy,mstd,xprd,yprds[ith],lp,xd,yds[ith],ld,
                         xd2,yd2s[ith],ld2,c_Theta,
                         Monotonic,Convex,"Sample",false,prePH,meanf,Rtt,Rpt,Rpp,Ktt0,Kpt0,Ktp0,Kpp0)
            #t3 = time()
            tAccept=tmp[1]
            if tAccept==true 
                tTheta = c_Theta
                tlogprior,tllh,tlogpost,tmuj,tSj=tmp[2],tmp[3],tmp[4],tmp[5],tmp[6]
                #tPH = tlogprior + tllh + tlogpost + tder + tder2
                achitT += 1
                iThetas[ith] = tTheta;
                logp0s[ith]=tlogprior; llhs[ith]=tllh; logposts[ith]=tlogpost
                mujs[ith]=tmp[5];Sjs[ith]= tmp[6]
            else 
                tmuj=mujs[ith];tSj=Sjs[ith]
            end

            # if ith == 1 
            #     s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" tPH "   llh:" tllh "   logpost:" tlogpost "   der:" tder "der2:" tder2
            #     println(s1,"@",tTheta)
            # end
            #t4 = time()

            c_yprd = updateyprd(yprds[ith],cqY,tSj)
            #c_yprd = updateyprd_m(yt[lt],ydel,lp,R,sigR)
            #c_yprd = updateyprd_n(typrd,lp,mstd)           

            #t5 = time()

            n_logpost=eval_logpost(c_yprd,tmuj,tSj)

            #t6 = time()

            n_der,n_der2 = eval_der(tstep,mstep,xt,yt,xprd,c_yprd,mstd,R,sigR)

            #t7 = time()
            prePH = tlogpost  + tder  + tder2
            newPH = n_logpost + n_der + n_der2
            yAccept=false
            if newPH - prePH > 0.0
                yAccept=true
            elseif log(rand()) <= newPH-prePH
                yAccept=true
            end
            if yAccept==true 
                tlogpost=n_logpost; tder = n_der; tder2 = n_der2
                typrd=c_yprd; achitY += 1
                yprds[ith]= c_yprd; yds[ith]=tyd; yd2s[ith]=tyd2
                logposts[ith]=tlogpost; ders[ith]=[tder,tder2]
            end
            tPH = tlogprior + tllh + tlogpost + tder + tder2
            PHs[ith]=tPH
            
            #t8 = time()
            #tsum = t8-t0
            #s1=@sprintf "%s %10.2e %s %8.1f %s %8.1f %s %8.1f %s %8.1f" "tsum" tsum "t1-0:" 100*(t1-t0)/tsum "t2-1:" 100*(t2-t1)/tsum "t3-2:" 100*(t3-t2)/tsum "t4-t3:" 100*(t4-t3)/tsum
            #s2=@sprintf "%s %8.1f %s %8.1f" " t5-4:" 100*(t5-t4)/tsum "t6-5:" 100*(t6-t5)/tsum 
            #s3=@sprintf "%s %8.1f %s %8.1f" " t7-6:" 100*(t7-t6)/tsum "t8-7:" 100*(t8-t7)/tsum
            #println(s1,s2,s3)
        end
        s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" PHs[1] "   llh:" llhs[1] "   logpost:" logposts[1] "   der:" ders[1][1] "der2:" ders[1][2]
        println(s1,"@",iThetas[1])
        #### ~Resampling~
        if  tstep == 200 || tstep == 1000            
            iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs = Resample(iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs,numN)
            cqY = qY * qYfac
        end
        ####### Adaptive proposals
        # if  500 > tstep > 200
        #     lqT=log(cqT);lqY=log(cqY)
        #     b_n = bn(0.1*tstep)
        #     lqT = lqT + b_n * (achitT/(numN*tstep) - rMH[1]) 
        #     lqY = lqY + b_n * (achitY/(numN*tstep) - rMH[2]) 
        #     #cqT = exp(lqT)
        #     cqY = exp(lqY)
        # end
        s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100*achitT/(numN*(tstep-1)) " Y:" 100*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
        println(s,"\n")
    end
    Pv=[ [ iThetas[ith], yprds[ith], yds[ith], yd2s[ith],
         PHs[ith], logp0s[ith], llhs[ith], logposts[ith], ders[ith][1],ders[ith][2]] for ith=1:numN]
    E0,Evar=Summary(mstep,numN,xt,yt,xun,yun,xprd,xd,oxt,oyt,Pv,muy,mstd,"Theta")
    Nmin=string(Int64(oxt[1])) ;  Nmax=string(Int64(oxt[length(oxt)]))
    if Monotonic==false && Convex == false
        fn="Thetas_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woMC.dat"
    elseif Monotonic==false && Convex == true
        fn="Thetas_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woM.dat"
    elseif Monotonic==true && Convex == false 
        fn="Thetas_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woC.dat"
    else
        fn="Thetas_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*".dat"
    end
    iot = open(fn, "w")
    for ith = 1:numN
        println(iot,Pv[ith][1][1]," ",Pv[ith][1][2]," ",Pv[ith][5], " ", Pv[ith][6]," ",Pv[ith][7] )
    end
    close(iot)
    return [E0,Evar]
end

function Mchole(tmpA::Array{Float64,2},ln::Int64) 
    cLL=cholesky(tmpA, Val(false); check = true).L
    logLii=0.0
    @simd for i = 1:ln
        logLii += log(cLL[i,i])
    end
    return inv(cLL), 2.0*logLii
end

function Simple_inv(tmpA::Array{Float64,2},ln::Int64) 
    try
        return  inv(tmpA),logdet(tmpA)
    catch
        try
            cLinv, logLii = Mchole(tmpA,ln)
            return transpose(cLinv)*cLinv,logLii
        catch
            errc = 0            
            while isposdef(tmpA)==false
                errc += 1
                minEV=minimum(eigvals(tmpA))
                if minEV==0.0
                    minEV=1.e-15
                end
                tmpA += Diagonal( 10.0*abs(minEV) * ones(Float64,ln,ln))
                if errc > 100;println("NotPSD:Sjoint: eigvals",eigvals(tmpA), " @ ",Theta, "isposdef(Sjoint)", isposdef(tmpA) );end
            end
            cLinv, logLii = Mchole(tmpA,ln)
            return transpose(cLinv)*cLinv,logLii
        end
    end
end

function nu_t(tstep,maxstep) 
    1.e+6* (tstep/maxstep)^0.1
    #1.e-2 * (10*x)**2 ### safe but slow
end

function weightedmean(x, w)
    wxsum = wsum = 0.0
    wsum2 = 0.0
    for (x,w) in zip(x,w)
        wx = w*x
        if !ismissing(wx)
            wxsum += wx
            wsum += w
            wsum2 += wx*x
        end
    end
    wstd=sqrt(wsum2 - wxsum*wxsum )
    return wxsum, wstd
end
function make_xyt(xall,yall,useind,l1,l2,multihw=false)
    xt=zeros(Float64,l1);yt=zeros(Float64,l1);xu=zeros(Float64,l2);yu=zeros(Float64,l2)    
    ysigma=zeros(Float64,l1)
    hit1=0;hit2=0
    #global hit1,hit2
    ndata=length(xall)
    if ndata != l1+l2 ; println("warn!!!");end        
    for i = 1:ndata
        if i in useind
            hit1 += 1 
            xt[hit1] = xall[i]
            if multihw==true 
                yt[hit1]=mean(yall[i])
                ysigma[hit1]=std(yall[i])
            else
                yt[hit1]=mean(yall[i])
            end
        else 
            hit2 += 1
            xu[hit2] = xall[i]
            if multihw==true
                yu[hit2]=mean(yall[i])
            else
                yu[hit2]=yall[i]
            end          
        end
    end
    if multihw == true 
        return xt,yt,ysigma,xu,yu
    else
        return xt,yt,xu,yu
    end
end
function ExtrapA(a,b,c,Nminv) 
    return a*exp(-b/Nminv)+c
end

function bn(tstep)
    return (tstep)^(-0.5)
end

function Phi(z)
    return 0.5 * erfc(-(z/(2^0.5)) )
end

function Summary(tstep,numN,xt,yt,xun,yun,xprd,xd,oxt,oyt,Pv,muy,mstd,plttype)   
    global bestPH,bestV,bestW, Wm, Ws
    lt=length(xt); lp=length(xprd); ld=length(xd)
    yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprd2s= [ [0.0 for ith=1:numN] for kk=1:lp]
    yds= [ [0.0 for ith=1:numN] for kk=1:ld]
    Weights  = [0.0 for ith=1:numN]    
    WeightsH = [0.0 for ith=1:numN]    
    Thetas=[[0.0 for ith=1:numN] for k=1:3]
    for ith=1:numN
        tmp=Pv[ith]
        tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2=tmp
        if length(tTheta)==2
            for kk=1:2 
                Thetas[kk][ith] = tTheta[kk]
            end
        else 
            for kk=1:3
                Thetas[kk][ith] = tTheta[kk]
            end
        end            
        for kk=1:length(xd)
            yds[kk][ith] = tyd[kk]
        end
        
        logw=tlogprior+tllh +tlogpost +tder + tder2 -100 ## -100:to avoid overflow
        logwH=tlogprior+tllh -50 ## to avoid overflow
        #Weights[ith] = logw;    WeightsH[ith] = logwH
        Weights[ith] = exp(logw);    WeightsH[ith] = exp(logwH)
        #        if ith==1; println(Weights[ith]);end
        for kk=1:lp
            yprds[kk][ith] = typrd[kk]*mstd.+muy
            w_yprds[kk][ith] = Weights[ith]*(typrd[kk]*mstd.+muy)
            w_yprd2s[kk][ith] = Weights[ith]*(typrd[kk]*mstd.+muy)*(typrd[kk]*mstd.+muy)
        end
        if ith==1
            bestV=[tTheta,typrd,tyd,tPH,tlogprior,tllh,tlogpost,tder,tder2]
            bestPH=-1.e+15;bestW=Weights[ith]
        elseif tPH > bestPH 
            #        elseif tllh > bestPH 
            bestV=[tTheta,typrd,tyd,tPH,tlogprior,tllh,tlogpost,tder,tder2]
            bestPH = tPH;bestW=Weights[ith]
        end                    
    end
    
    sumW=sum(Weights)
    sumWH=sum(WeightsH)
    w_yprds = w_yprds/sumW
    w_yprd2s = w_yprd2s/sumW       
    
    means=[ 0.0 for i=1:lp]
    stds=[ 0.0 for i=1:lp]
    for kk=1:lp
        tmean = sum(w_yprds[kk])
        if sum(w_yprd2s[kk]) - tmean*tmean < 0. 
            tstd  = 1.e-6
        else
            tstd  = sqrt(sum(w_yprd2s[kk]) - tmean*tmean )
        end
        means[kk]=tmean;stds[kk]=tstd
    end    
    # if plttype!="R"
    #     println("Best Thetas:",bestV[1]," PH ",bestV[4]," llh= ",bestV[6]," logpost ",bestV[7]," logder ",bestV[8]," W ",bestW/sumW)
    # end
    
    bestT,besty,bestyd=bestV[1],bestV[2]*mstd.+muy,bestV[3]*mstd.+muy   
    Oyun = yun
    
    if tstep  == mstep
        Nmin=string(Int64(oxt[1])) 
        Nmax=string(Int64(oxt[length(oxt)]))
        if Monotonic==false && Convex == false
            fn="Posterior_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woMC.dat"
        elseif Monotonic==false && Convex == true
            fn="Posterior_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woM.dat"
        elseif Monotonic==true && Convex == false 
            fn="Posterior_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*"_woC.dat"
        else
            fn="Posterior_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*".dat"
        end
        io = open(fn, "w")
        println(io,inpname)
        println(io,"xprd=")
        println(io,xprd)
        println(io,"\n means=")
        println(io,means)
        println(io,"\n stds=")
        println(io,stds)
        println(io,"#weights")
        for ith =1:numN
            s=@sprintf "%16.10e" Weights[ith]
            print(io, s)
            for j = 1: length(means)
                s=@sprintf " %12.6f " yprds[j][ith] 
                print(io,s)
            end
            print(io,"\n")
        end
        close(io)
    end
    
    if CImode=="NCSM" && plttype=="R" && tstep == Rstep
        deno=(yt[lt-1]-yt[lt])*mstd
        nume=yt[lt]*mstd+muy .- yprds[1]
        println("yt[lt-1,lt]", yt[lt-1]*mstd+muy, " ", yt[lt]*mstd+muy, " yprd ",means[lp])                
        Dratio = nume ./deno
        Wm,Ws=weightedmean(Dratio,Weights/sumW)
        println("R(mean)=$Wm, Rstd=$Ws")
        # fig = figure(figsize=(4,4))
        # ax = fig.add_subplot(111)
        # xlabel("\$ y(Nmax+2) \$ ") 
        # ax.set_facecolor("#D3DEF1")
        # testl=collect(-10:0.01:10)
        # ax.set_xlim(0.2,1.0)
        # ax.hist(Dratio,bins=testl,weights=Weights/sumW,color="orange",alpha=0.3)
        # s= @sprintf "%10.3f %s %8.3f" Wm "+-" Ws
        # ax.text(0.2,0.8,s,transform=ax.transAxes)
        # tdir=string(inttype)
        # savefig("pic/yratio_"*string(Kernel)*".pdf")
        # close()        
        return Wm, Ws,means[lp]
    end       
    return means[lp],stds[lp]
end

function calcSj(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                Kpt::Array{Float64,2},Kpp::Array{Float64,2},
                yt::Array{Float64,1},mu_yt::Array{Float64,1},mu_yp::Array{Float64,1},
                tKtp::Array{Float64,2},tKpp::Array{Float64,2})
    mul!(tKtp,cLinv,transpose(Kpt))    
    mul!(tKpp,transpose(tKtp),tKtp)
    return mu_yp + Kpt*(transpose(cLinv)*(cLinv*(yt-mu_yt))), Kpp - tKpp
end

function eval_der_sigma(tstep,mstep,yprd::Array{Float64,1},yd::Array{Float64,1},yd2::Array{Float64,1},Monotonic,Convex)
    ### for derivative obs.
    der = 0.0; der2 = 0.0
    nu  = nu_t(tstep,mstep)
    @simd for kk=1:length(yd)
        tmp = Phi(nu*yd[kk])
        if tmp == 0.0 ; der += - 1.e+10
        else ; der += log(tmp);end
    end
    if Convex==true 
        convcost=nu #*1.e+1
        for kk=1:length(yd2)
            tmp = Phi(convcost*yd2[kk])
            if tmp == 0.0;der2 += -1.e+10
            else ; der2 += log(tmp);end
        end
    end 
    return der, der2
end

function eval_der(tstep,mstep,xt::Array{Float64,1},yt::Array{Float64,1},xprd::Array{Float64,1},yprd::Array{Float64,1},mstd,R,sigR)
    der = 0.0 ; der2 = 0.0
    nu  = nu_t(tstep,mstep)
    lp=length(yprd);lt=length(yt)
    nupen=1.e+10
    tmp = Phi( nu * (yt[lt]-yprd[1])*mstd)
    if tmp == 0.0
        der += -nupen
    else 
        der += log(tmp)
    end
    if lp > 1
        @simd for k =1:lp-1
            tmp = Phi( nu*(yprd[k]-yprd[k+1])*mstd)
            if tmp == 0.0
                der += -nupen
            else 
                der += log(tmp)
            end
        end
    end

    convcost = nu 
    convpen = 1.e+8
        
    if xprd[1]> 0 
        tR = (yt[lt]-yprd[1])/(yt[lt-1]-yt[lt])
        U_idealR=R+sigR; L_idealR=R-sigR
        #### Non Gaussian 
        tmp = Phi( convcost * (U_idealR-tR) ) #+ Phi( convcost * (tR-L_idealR) )     
        if tmp == 0.0
            der2 += -convpen 
        else 
            der2 += log(tmp)
        end            
        ### Gaussian 
        ##der2 += - 0.5 * (tR-R)^2 / (sigR*sigR)                
        if lp > 1 
            tR = (yprd[1]-yprd[2])/(yt[lt]-yprd[1]) 
            Nk=Int64(xprd[2]); Nj=Int64(xprd[1])
            Nt=Int64(xt[lt]) ; Ni=Nt
            
            if abs(Nj-Nt) == 2 && abs(Nk-Nj)==2
                tmp = Phi( convcost * (U_idealR-tR) ) #+ Phi( convcost * (tR-L_idealR) )     
            else
                tmp = 1.0
            end
            if tmp == 0.0
                der2 += -convpen
            else 
                der2 += log(tmp)
            end                 
            
            if lp>2 
                for k =1:lp-2 ### increment 2                 
                    if abs((yprd[k+1]-yprd[k+2])*mstd) < 1.e-4 ;continue;end
                    tR = (yprd[k+1]-yprd[k+2])/(yprd[k]-yprd[k+1])
                    #tR = (yprd[k+1]-yprd[k+2])/(yprd[k]-yprd[k+1]); U_idealR= (R + sigR)^2                   
                    tmp = Phi( convcost * (U_idealR-tR) )
                    if tmp == 0.0
                        der2 += -convpen 
                    else 
                        der2 += log(tmp)
                    end
                end
            end
        end
    end

    if Monotonic == false;der = 0.0;end
    if Convex == false ;der2 = 0.0;end
    return der, der2
end


function eval_logpost(yprd::Array{Float64,1},mujoint::Array{Float64,1},Sjoint::Array{Float64,2})
    Sjinv,logdetSj = Simple_inv(Sjoint,length(mujoint)) ## 90% time of eval_logpost 
    return -0.5 * transpose(yprd-mujoint)*(Sjinv*(yprd-mujoint))  - 0.5 * logdetSj
end

function evalPH_t(Kernel,tstep,mstep,ith,xt::T,yt::T,lt,
                  Mysigma,muy,mstd,xprd::T,yprd::T,lp,xd::T,yd::T,ld,
                  xd2::T,yd2::T,ld2,Theta,
                  Monotonic,Convex,GPmode,fixR,prePH,meanf,Rtt::T2,Rpt::T2,Rpp::T2,Ktt0::T2,Kpt0::T2,Ktp0::T2,Kpp0::T2) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    #time0 = time()

    # if Kernelder == true ## Not complete
    #     KernelMat!(Ktt,Kernel,Theta,xt,xt,0,false) + Diagonal( (1.e-8) * ones(Float64,lt,lt))
    #     KernelMat!(Kpp,Kernel,Theta,xprd,xprd,0,false) + Diagonal( (1.e-8) * ones(Float64,lp,lp))
    #     KernelMat!(Kpt,Kernel,Theta,xprd,xt,0,false) 
    #     KernelMat!(Kdd,Kernel,Theta,xd,xd,2,false) + Diagonal( (1.e-8) * ones(Float64,ld,ld))
    #     KernelMat!(Kpd,Kernel,Theta,xprd,xd,1,false) 
    #     KernelMat!(Ktd,Kernel,Theta,xt,xd,1,false) 
    #     cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))        
    #     if Convex==true 
    #         K02=KernelMat!(Kernel,Theta,xt,xd2,3,false) 
    #         Kp2=KernelMat!(Kernel,Theta,xprd,xd2,3,false) 
    #         K12=KernelMat!(Kernel,Theta,xd,xd2,4,false) 
    #         ld2=length(xd2)
    #         K22=KernelMat!(Kernel,Theta,xd2,xd2,5,false) + Diagonal( (1.e-4) * ones(Float64,ld2,ld2))
    #         A0 = hcat(hcat(Kpp,Kpd),Kp2)
    #         A1 = hcat(hcat(transpose(Kpd),Kdd),K12)
    #         A2 = hcat(hcat(transpose(Kp2),transpose(K12)),K22)
    #         A  = vcat(vcat(A0,A1),A2)
    #         C  = vcat(vcat(Kpt,transpose(Ktd)),transpose(K02)) 
    #     else
    #         A0 = hcat(Kpp,Kpd)
    #         A1 = hcat(transpose(Kpd),Kdd)
    #         A  = vcat(A0,A1)
    #         C  = vcat(Kpt,transpose(Ktd))
    #     end

    #     mu_y = reshape([ 0.0 for i=1:lp],lp)
    #     mu_y = reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i]) for i =1:lp],lp)
    #     mu_yt= reshape([ 0.0 for i=1:lt],lt)
    #     mu_d = reshape([ 0.0 for i=1:ld],ld)
    #     if Convex==true
    #         mu_d2= reshape([ 0.0 for i=1:length(xd2)],length(xd2))
    #         mujoint = vcat(vcat(mu_y,mu_d),mu_d2)
    #     else
    #         mujoint = vcat(mu_y,mu_d)
    #     end
    #     mujoint += C*Kttinv*yt        
    #     tmp=cLinv*transpose(C)
    #     Sjoint = A - transpose(tmp)*tmp
    #     errc = 0
    #     while isposdef(Sjoint)==false
    #         errc += 1
    #         ttmp=eigvals(Sjoint)
    #         minEV=minimum(ttmp)
    #         if minEV==0.0
    #             minEV=1.e-15
    #         end
    #         Sjoint += Diagonal( 10*abs(minEV) * ones(Float64,length(mujoint),length(mujoint)))
    #         if errc > 100;println("NotPSD:Sjoint: eigvals",eigvals(Sjoint), " @ ",Theta, "isposdef(Sjoint)", isposdef(Sjoint) );end
    #     end
    # else
    #t0_0 = time()        
    Ktt,Kpt,Kpp = KernelMat(Kernel,Theta,xt,lt,xprd,lp,Rtt,Rpt,Rpp,false)            
    #t0_1 = time()

#    Ktt,Kpt,Kpp = KernelMat2(Kernel,Theta,xt,lt,xprd,lp,Rtt,Rpt,Rpp,false) 
#    tm1 = time()

#    println("time  (KernelMat) ",t0_1-t0_0)
#    println("time (KernelMat2) ",tm1-t0_1)
    #if isposdef(Ktt) == false;println("issymmetric(Ktt)",issymmetric(Ktt), " @Theta=", Theta);println("eigvals:",eigvals(Ktt));end
    
    cLinv,logdetK = Mchole(Ktt,lt)
    #Kttinv,logdetK = Simple_inv(Ktt)        
    
    #t0_2 = time()
    mu_yt = meanf[1:lt]
    mu_yp = meanf[lt+1:lt+lp]
    mujoint,Sjoint  = calcSj(cLinv,Kpt,Kpp,yt,mu_yt,mu_yp,Ktp0,Kpp0)
    #t0_3 = time()
    #tsum= t0_3-t0_0
    #s1=@sprintf "Time for KernelMat  %s %10.2e %s %8.1f %s %8.1f" "tsum" tsum "t1-0:" 100*(t0_1-t0_0)/tsum "t2-1:" 100*(t0_2-t0_1)/tsum 
    #s2=@sprintf " %s %8.1f" "t3-2:" 100*(t0_3-t0_2)/tsum 
    #println(s1,s2,"\n")
    
    #time1 = time()

    logprior = 0.0 ## uniform     
    term1 = -0.5*transpose(yt-mu_yt)*(transpose(cLinv)*(cLinv*(yt-mu_yt)))
    term2 = -0.5*logdetK
    llh = term1 + term2 

    #time2 = time()

    Accept =false
    logpost= eval_logpost(yprd,mujoint,Sjoint)
    if tstep == 1
        der,der2 = eval_der(tstep,mstep,xt,yt,xprd,yprd,mstd,R,sigR)
        PH = logprior + llh + logpost + der + der2
        return Accept,PH,logprior,llh,logpost,der,der2,mujoint,Sjoint
    end
    
    #time3 = time()
    #tsum = time3-time0
    #s1=@sprintf "Time for evalPH %s %10.2e %s %8.1f %s %8.1f %s %8.1f " "tsum" tsum "t1-0:" 100*(time1-time0)/tsum "t2-1:" 100*(time2-time1)/tsum "t3-2:" 100*(time3-time2)/tsum
    #println(s1)

    if logprior+llh+logpost > prePH               
        return true,logprior,llh,logpost,mujoint,Sjoint
    elseif log(rand()) < logprior+llh+logpost-prePH
        return true,logprior,llh,logpost,mujoint,Sjoint
    else
        return [false]
    end
end

function logMat(tau::Float64,theta_r::Float64)
    tau * (1 + theta_r + theta_r^2 /3) * exp(-theta_r)
end

function KernelMat(ktype,Theta,xt::T,lt,xp::T,lp,Rtt::T2,Rpt::T2,Rpp::T2,pder=false) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    Ktt = zeros(Float64,lt,lt); Kpt = zeros(Float64,lp,lt); Kpp = zeros(Float64,lp,lp)
    tau,sigma=Theta
    theta = sqrt(5.0)/sigma
    @inbounds @simd for j=1:lt
        for i=j:lt
            tmp  = logMat(tau,theta*Rtt[i,j])
            Ktt[i,j] = tmp; Ktt[j,i] = tmp 
        end
        for i=1:lp
            Kpt[i,j] = logMat(tau,theta*Rpt[i,j])
        end
    end
    @inbounds @simd for j=1:lp
        for i=j:lp
            tmp  = logMat(tau,theta*Rpp[i,j])
            Kpp[i,j] = tmp; Kpp[j,i] = tmp
        end
    end
    return Ktt,Kpt,Kpp
end
