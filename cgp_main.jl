function B3fit(xt,yt)
    lt = length(xt)
    xfit,yfit = xt[lt-2:lt], yt[lt-2:lt]
    @. model(x, p) = p[1] + p[2]*exp(-x*p[3])
    p0 = [-35.0, 1.0, 1.0]
    ub = [yt[lt], Inf, Inf]
    fit = curve_fit(model, xfit, yfit, p0)
    pfit = coef(fit) 
    return pfit
end

function log_d(xi::Float64,xj::Float64)
    abs(log(xi)-log(xj))  
end

function raw_d(xi::Float64,xj::Float64)
    abs(xi-xj)  
end

function makeRm(xt,xp,lt,lp,ktype)
    Rtt=zeros(Float64,lt,lt);Rpt=zeros(Float64,lp,lt);Rpp=zeros(Float64,lp,lp)
    if occursin("log",Kernel)
        tf = log_d
    else
        tf = raw_d
    end
    for i=1:lt
        for j=1:lt
            Rtt[i,j]=tf(xt[i],xt[j])
        end
    end
    for i=1:lp
        for j=1:lt
            Rpt[i,j]=tf(xp[i],xt[j])
        end
    end
    for i=1:lp
        for j=1:lp
            Rpp[i,j]=tf(xp[i],xp[j])
        end
    end
    return Rtt,Rpt,Rpp
end

function Resample(iThetas, yprds, yds, yd2s,
                  PHs, logp0s, llhs, logposts,
                  ders, mujs, Sjs,numN::I
                  ) where {T<:Array{Float64,1},T2<:Array{Float64,2},I<:Int64,F<:Float64}
    w_der=[0.0 for i=1:numN]
    x=[ [] for i=1:numN]
    Pv = [ [iThetas[ith], yprds[ith], yds[ith], yd2s[ith],
            PHs[ith], logp0s[ith], llhs[ith], logposts[ith],
            ders[ith], mujs[ith], Sjs[ith]] for ith=1:numN]
    for i =1:numN
        tmp= ders[i][1] + ders[i][2]
        if tmp > 709.0
            w_der[i]= 1.e+80
        elseif tmp<-746.0
            w_der[i]=1.0e-80
        else
            w_der[i]=exp(tmp)
        end
    end
    StatsBase.alias_sample!(Pv,weights(w_der),x)
    iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs= [ [x[ith][jj] for ith=1:numN] for jj=1:11]
    return iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs
end                    

function readinput(inpname,xpMax,Monotonic,Convex,ktype)
    tmp=split(inpname,"_")[(length(split(inpname,"_")))]
    Kernel=string(ktype)
    if Kernel=="logMat52" || Kernel=="logRBF"
        Tsigma = [1.0 0.0;0.0 1.0];tTheta=[5.0,5.0]
    else    
        Tsigma = [0.1 0.0;0.0 0.1];tTheta=[1.0,1.0]
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
    println("x(data) $oxt")
    xt=oxt;yt=oyt
    olt=length(xt)
    pfit=B3fit(xt,yt)
    if Kernel=="logRBF" || Kernel=="logMat52"
        iThetas=[ [100.0*rand(), 1.0*rand()] for i=1:numN]
    else
        iThetas=[ [100.0*rand(),5.0*rand()] for i=1:numN]       
    end
   
    lt=length(xt)
    Ev=0.0
    muy=minimum(yt)
    mstd=std(yt)
    xprd=Float64[];pNmax=[]

    if Monotonic ==true
        if Convex==true
            unx=collect(xt[lt]+2:2.0:xpMax)
        else
            unx=collect(xt[lt]+2:4.0:xpMax)
        end
    else
        unx=collect(xt[lt]+2:4.0:xpMax)
    end

    for tmp in unx 
        if (tmp in xt)==false 
            push!(xprd,tmp)
        end
    end
    lp=length(xprd)
    Rms=makeRm(xt,xprd,lt,lp,ktype)
    return Tsigma,tTheta,xt,yt,xprd,xun,yun,oxt,oyt,iThetas,lt,lp,muy,mstd,pfit,Rms
end

function proposal_y(typrd::Array{Float64,1},cqY::Float64,Sjoint::Array{Float64,2})
    try
        rand(MvNormal(typrd,cqY*Sjoint))
    catch
        typrd .+1.e+1
    end
end

function Gibbs_yprd(yprd::Array{Float64,1},cqY::Float64,tstep::Int64,mstep::Int64,ind::Int64)
    ty = copy(yprd)
    ty[ind] += rand( Normal(0.0, cqY) )
    return ty
end

function updateTheta(tTheta::Array{Float64,1},sigma_T::Array{Float64,2})
    exp.(rand(MvNormal(log.(tTheta),sigma_T)))
end

function y_Gibbs(yprd::T,cqY::F,tSj::T2,tmuj::T,tTheta::T,lp::I,
                 tstep::I,mstep::I,mstd::F,R::F,sigR::F,
                 xt::T,yt::T,xprd::T,tlogpost::F,tder::F,
                 tder2::F,cLp::L,Monotonic,Convex
                 ) where {T<:Array{Float64,1},
                          T2<:Array{Float64,2},
                          I<:Int64,F<:Float64,
                          L<:LowerTriangular{Float64,Array{Float64,2}}}
    typrd = copy(yprd)    
    acchit = 0.0
    @inbounds for kplace = 1:lp
        c_yprd = Gibbs_yprd(typrd,cqY,tstep,mstep,kplace)
        n_logpost=eval_logpost(c_yprd,tmuj,tSj,tTheta,lp,cLp)
        n_der,n_der2 = eval_der(tstep,mstep,xt,yt,xprd,c_yprd,mstd,R,sigR,Monotonic,Convex)
        diff = n_logpost + n_der + n_der2 -(tlogpost +  tder + tder2)
        if diff > 0
            tlogpost=n_logpost; tder = n_der; tder2 = n_der2            
            acchit += 1.0;typrd= c_yprd        
        elseif log(rand()) <= diff
            tlogpost=n_logpost; tder = n_der; tder2 = n_der2            
            acchit += 1.0;typrd= c_yprd        
        end
    end
    return typrd,tlogpost, tder,tder2, acchit/lp
end

function y_corr(yprd::T,cqY::F,tSj::T2,tmuj::T,tTheta::T,lp::I,
                tstep::I,mstep::I,mstd::F,R::F,sigR::F,
                xt::T,yt::T,xprd::T,tlogpost::F,tder::F,
                tder2::F,cLp::L,Monotonic,Convex
                ) where {T<:Array{Float64,1},
                         T2<:Array{Float64,2},
                         I<:Int64,F<:Float64,
                         L<:LowerTriangular{Float64,Array{Float64,2}}}
    c_yprd = proposal_y(yprd,cqY,tSj)
    n_logpost=eval_logpost(c_yprd,tmuj,tSj,tTheta,lp,cLp)    
    n_der,n_der2 = eval_der(tstep,mstep,xt,yt,xprd,c_yprd,mstd,R,sigR,Monotonic,Convex)
    diff = n_logpost + n_der + n_der2 - tlogpost - tder - tder2 
    achit=0.0    
    if log(rand()) <= diff
        return c_yprd,n_logpost, n_der,n_der2, 1.0
    else
        return yprd,tlogpost, tder,tder2, 0.0
    end

end

function main(mstep::Int64,numN::Int64,sigRfac::Float64,
              ktype,updatefunc,inpname,inttype,
              xpMax,Monotonic,Convex,paramean,qT,qY,qYfac)
    Tsigma,tTheta,xt,yt,xprd,xun,yun,oxt,oyt,iThetas,lt,lp,muy,mstd,pfit,Rms=readinput(inpname,xpMax,Monotonic,Convex,ktype)
    Rtt,Rpt,Rpp=Rms
    xd=[0.0]; xd2=[0.0]
    ld=length(xd);ld2=length(xd2)
    yprd=0*xprd.-1.e-2; yd=zeros(Float64,ld);  yd2=zeros(Float64,ld2)

    R=(yt[lt-1]-yt[lt])/(yt[lt-2]-yt[lt-1]);sigR = sigRfac * R
    println("xprd $xprd R $R sigR $sigR")
    
    achitT=0.;achitY=0.
    cqT=qT;cqY=qY
    if paramean==true
        mu_yt  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xt[i])   for i =1:lt],lt).-muy)/mstd
        mu_yp  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i]) for i =1:lp],lp).-muy)/mstd
    else 
        mu_yt = reshape([ 0.0 for i =1:lt],lt); mu_yp = reshape([ 0.0 for i =1:lp],lp)
    end

    tmuj=zeros(Float64,lp+ld+ld2);tSj=ones(Float64,lp+ld+ld2,lp+ld+ld2)
    tlogprior=-1.e+50;tllh=-1.e+50;tlogpost=-1.e+50;tder=-1.e+50;tder2=-1.e+50
    tPH =  tlogprior + tllh + tlogpost + tder + tder2
    yprds= [ yprd for i=1:numN]; yds=[yd for i=1:numN];yd2s=[yd for i=1:numN]
    PHs = [ tPH for i=1:numN]; logp0s = [tlogprior for i=1:numN]
    llhs=[tllh for i=1:numN]; logposts=[tlogpost for i=1:numN]; ders=[ [tder,tder2] for i=1:numN]
  
    yt= (yt .-muy)/mstd;   ydel = yt[lt-1]-yt[lt]   

    Ktt = zeros(Float64,lt,lt); Kpp = zeros(Float64,lp,lp)
    Kpt = zeros(Float64,lp,lt); Ktp = zeros(Float64,lt,lp)
    muj0 = zeros(Float64,lp); Sj0 =zeros(Float64,lp,lp)
    mujs=[ muj0 for i=1:numN]; Sjs =[ Sj0 for i=1:numN]
    cLL = LowerTriangular(zeros(Float64,lt,lt))
    cLp = LowerTriangular(zeros(Float64,lp,lp))
    if paramean==true;NDist=Normal(0,0.001);else; NDist=Normal(R,sigR);end
    tstep = 1
    for ith=1:numN 
        #tyd=0*xd; tyd2=0*xd2
        tmp= evalPH_t(ktype,tstep,mstep,ith,xt,yt,lt,muy,
                      mstd,xprd,yprd,lp,iThetas[ith],
                      Monotonic,Convex,-1.e+10,mu_yt,mu_yp,
                      Rtt,Rpt,Rpp,Ktt,Kpt,Ktp,Kpp,R,sigR,mujs[ith],Sjs[ith],cLL,cLp)
        if paramean==true
            pn=sign( 0.5-rand())
            c_yprd= [ mu_yp[i] + pn*0.01*mu_yp[i] for i=1:lp ]
        else
            rtmp = rand(NDist)
            c_yprd= [ yt[lt] - ydel * rtmp * (rtmp^(i)-1.0)/(rtmp-1.0) for i=1:lp]
        end
        
        if length(tmp) > 6
            Accept,tlogprior,tllh,tlogpost,tder,tder2,tmuj,tSj = tmp
        else
            tder,tder2 = eval_der(tstep,mstep,xt,yt,xprd,c_yprd,mstd,R,sigR,Monotonic,Convex)         
            tlogprior = 0.0; tllh = -1.e+50; tlogpost = -1.e+50;
        end
        tPH =  tlogprior+tllh+tlogpost+tder+tder2
        yprds[ith]= c_yprd#; yds[ith]=tyd; yd2s[ith]=tyd2
        PHs[ith]=tPH; logp0s[ith]=tlogprior; llhs[ith]=tllh; logposts[ith]=tlogpost
        ders[ith]=[tder,tder2];mujs[ith]=tmuj;Sjs[ith]= tSj
    end
    
    sigma_T = cqT*Tsigma
    c_yd=0.0*xd;c_yd2=xd2*0.0   #### to be changed if derKer introduced         
    for tstep=2:mstep
        if tstep % 1000 == 0;println("tstep ",tstep);end
        #println("tstep ",tstep)
        @inbounds @simd for ith=1:numN 
            #t0 =time()  ###
            tTheta=copy(iThetas[ith])
            tder,tder2=ders[ith]
            tPH,tlogprior,tllh,tlogpost = PHs[ith],logp0s[ith], llhs[ith],logposts[ith]
            prePH = tlogprior + tllh + tlogpost
            #t1 = time()  ###
            c_Theta = updateTheta(tTheta,sigma_T)
            
            #t2 = time()  ###
            tmp=evalPH_t(ktype,tstep,mstep,ith,xt,yt,lt,
                         muy,mstd,xprd,yprds[ith],lp,c_Theta,
                         Monotonic,Convex,prePH,mu_yt,mu_yp,
                         Rtt,Rpt,Rpp,Ktt,Kpt,Ktp,Kpp,R,sigR,muj0,Sj0,cLL,cLp)
            #t3 = time()  ###
            if tmp[1]              
                tTheta = copy(c_Theta)
                iThetas[ith] = copy(tTheta)
                tlogprior,tllh,tlogpost=tmp[2],tmp[3],tmp[4]
                achitT += 1.
                mujs[ith]=copy(muj0); Sjs[ith]= deepcopy(Sj0)
            end
            #t4 = time()  ###
            c_yprd,tlogpost,tder,tder2,thit = updatefunc(
                yprds[ith],cqY,Sjs[ith],mujs[ith],tTheta,lp,
                tstep,mstep,mstd,R,sigR,
                xt,yt,xprd,tlogpost,tder,tder2,cLp,Monotonic,Convex)
            yprds[ith] = copy(c_yprd)
            logp0s[ith]=tlogprior
            llhs[ith]=tllh
            logposts[ith]=tlogpost
            ders[ith]=[tder,tder2]
            PHs[ith] = tlogprior + tllh + tlogpost + tder +tder2
            achitY += thit
            # t5 = time()  ###
            # tsum = t5-t0
            # s1=@sprintf "%s %10.2e %s %8.1f %s %8.1f" "tsum" tsum "t1-0:" 100*(t1-t0)/tsum "t2-1:" 100*(t2-t1)/tsum 
            # s2=@sprintf "%s %8.1f %s %8.1f %s %8.1f"  "t3-2:" 100*(t3-t2)/tsum "t4-t3:" 100*(t4-t3)/tsum  " t5-4:" 100*(t5-t4)/tsum
            # println(s1,s2)
        end
        #### ~Resampling~
        if  (tstep == 200 || tstep == 500 ) && (Monotonic==true || Convex ==true)
            iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs = Resample(iThetas, yprds, yds, yd2s, PHs, logp0s, llhs, logposts, ders, mujs, Sjs,numN)
            if tstep == 200
                cqY = qY * qYfac
            end
            if tstep == 500
                if 100.0*achitY/(numN*(tstep-1)) < 5.0
                    cqY = 0.05 *cqY
                elseif 100.0*achitY/(numN*(tstep-1)) < 13.0
                    cqY = 0.5 *cqY
                elseif 100.0*achitY/(numN*(tstep-1)) > 55.0
                    cqY = 100.0 * cqY
                elseif 100.0*achitY/(numN*(tstep-1)) > 35.0
                    cqY = 10.0 * cqY
                end
            end
        end       
        ####### Adaptive proposals
        # if  500 > tstep > 200
        #     #lqT=log(cqT);
        #     lqY=log(cqY)
        #     b_n = bn(0.1*tstep)
        #     #lqT = lqT + b_n * (achitT/(numN*tstep) - 0.30) 
        #     lqY = lqY + b_n * (achitY/(numN*tstep) - 0.30) 
        #     #cqT = exp(lqT)
        #     cqY = exp(lqY)
        # end
        # s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" PHs[1] "llh:" llhs[1] "logpost:" logposts[1] " der:" ders[1][1] "der2:" ders[1][2]
        # println(s1,"@ith=1,Thetas=",iThetas[1],"\n")
        if tstep %1000 == 0
            s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100.0*achitT/(numN*(tstep-1)) " Y:" 100.0*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
            println(s)
        end
        #s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100.0*achitT/(numN*(tstep-1)) " Y:" 100.0*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
        #println(s,"\n")
    end
    Pv=[ [ iThetas[ith], yprds[ith], yds[ith], yd2s[ith],
           PHs[ith], logp0s[ith], llhs[ith], logposts[ith],
           ders[ith][1],ders[ith][2],mujs[ith],Sjs[ith]]  for ith=1:numN]
    E0,Evar=Summary(mstep,numN,xt,yt,xun,yun,xprd,xd,oxt,oyt,Pv,muy,mstd,"Theta",inpname,inttype,Monotonic,Convex,paramean)
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

function AAt(A::Array{Float64,2})
    A*transpose(A)
end
function AtA(A::Array{Float64,2})
    transpose(A) * A
end

function VtV(V::Array{Float64,1})
    transpose(V) * V
end

function LtL(L::LowerTriangular{Float64,Array{Float64,2}})
    transpose(L) * L
end

function Mchole(tmpA::Array{Float64,2},
                ln::Int64,cLL::LowerTriangular{Float64,Array{Float64,2}})
    try 
        cLL=cholesky(tmpA).L
        logLii=0.0
        @inbounds @simd for i = 1:ln
            logLii += log(cLL[i,i])
        end
        return inv(cLL), 2.0*logLii
    catch
        return false, -1.e+100
    end
end

function Mcholeinv(tmpA::Array{Float64,2},ln::Int64)
    cLL=cholesky(tmpA).L
    logLii=0.0
    @inbounds @simd for i = 1:ln
        logLii += log(cLL[i,i])
    end
    return LtL(inv(cLL)), 2.0*logLii
end

function nu_t(tstep,maxstep) 
    1.e+6 * (tstep/maxstep)^0.1
    #1.e-3 * (10.0*tstep)^2.0 
    #1.e+1 * (10.0*tstep)^2.0 
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
    return 0.5 * erfc(-(z/(2.0^0.5)) )
end

function Summary(tstep,numN,xt,yt,xun,yun,xprd,xd,oxt,oyt,
                 Pv,muy,mstd,plttype,inpname,inttype,
                 Monotonic,Convex,paramean)   
    global bestV,bestW, Wm,Ws,bestmuj,bestSj
    lt=length(xt); lp=length(xprd); ld=length(xd)
    yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprd2s= [ [0.0 for ith=1:numN] for kk=1:lp]
    yds= [ [0.0 for ith=1:numN] for kk=1:ld]
    Weights  = [0.0 for ith=1:numN]    
    WeightsH = [0.0 for ith=1:numN]
    Thetas=[[0.0 for ith=1:numN] for k=1:3]
    bestPH= -1.e+15
    bestmuj = zeros(Float64,lp); bestSj = zeros(Float64,lp,lp)
    PHs  = [0.0 for ith=1:numN]    
    for ith=1:numN
        tmp=Pv[ith]
        tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmuj,tSj=tmp
        PHs[ith] = tPH
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
        #println("tPH $tPH tlogprior $tlogprior tllh $tllh tlogpost $tlogpost")
        logw=tlogprior+tllh +tlogpost +tder + tder2 -50.0 ## -100:to avoid overflow
        logwH=tlogprior+tllh -50.0 ## to avoid overflow
        #Weights[ith] = logw;    WeightsH[ith] = logwH
        Weights[ith] = exp(logw);    WeightsH[ith] = exp(logwH)
        if abs(tPH-(tlogprior + tllh +tlogpost +tder + tder2)) > 1.e-6
            println("error in tPH $tPH $tlogprior $tllh $tlogpost $tder $tder2")
        end
        for kk=1:lp
            yprds[kk][ith] = typrd[kk]*mstd.+muy
            w_yprds[kk][ith] = Weights[ith]*(typrd[kk]*mstd.+muy)
            w_yprd2s[kk][ith] = Weights[ith]*(typrd[kk]*mstd.+muy)*(typrd[kk]*mstd.+muy)
        end
        if tPH > bestPH 
            bestV=[tTheta,typrd,tyd,tPH,tlogprior,tllh,tlogpost,tder,tder2]
            bestPH = tPH;bestW=Weights[ith]
            bestmuj = tmuj ; bestSj =tSj
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

    
    bestT,besty,bestyd=bestV[1],bestV[2]*mstd.+muy,bestV[3]*mstd.+muy
    # println("bestPH ", bestPH, " llh= ",bestV[6], " logpost ", bestV[7], " logder ",bestV[8], " W ", bestW/sumW)
    # println("Theta ", bestV[1]); println("yprd $besty")
    # println(" muj ",bestmuj*mstd.+muy)
    Oyun = yun
    cLp = LowerTriangular(zeros(Float64,lp,lp)) 
    # println("eval_post", eval_logpost(bestV[2],bestmuj,bestSj,bestT,lp,cLp))
    # if isposdef(bestSj)
    #     cLinv,logdetSj=Mchole(bestSj,lp,cLp)
    #     term1 = -0.5* VtV( cLinv*(bestV[2]-bestmuj) )
    #     term2 = - 0.5*logdetSj
    #     println("term1 $term1 term2 $term2")
    # end

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
            s=@sprintf "%16.10e" PHs[ith]
            print(io, s)
            for j = 1: length(means)
                s=@sprintf " %12.6f " yprds[j][ith] 
                print(io,s)
            end
            print(io,"\n")
        end
        close(io)
    end
    return means[lp],stds[lp]
end

function calcmuj!(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                 Kpt::T2,Kpp::T2,yt::T,mu_yt::T,mu_yp::T,lp::Int64,
                 mujoint::T) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    tmp =  Kpt*(transpose(cLinv)*(cLinv*(yt-mu_yt)))
    @inbounds @simd for i =1:lp
        mujoint[i] = mu_yp[i] + tmp[i]
    end
    return nothing
end

function calcSj!(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                 Kpt::T2,Kpp::T2,lp::Int64,Sjoint::T2
                 ) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    tmp = - AAt(Kpt * transpose(cLinv))
    @inbounds @simd for j = 1:lp
        for i = 1:lp
            Sjoint[i,j] = Kpp[i,j] + tmp[i,j]
        end
    end
    nothing
end

function calcmuj(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                 Kpt::T2,Kpp::T2,yt::T,mu_yt::T,mu_yp::T,lp::Int64
                 ) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    mu_yp + Kpt*(transpose(cLinv)*(cLinv*(yt-mu_yt)))
end
function calcSj(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                 Kpt::T2,Kpp::T2,lp::Int64
                 ) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    Kpp - AAt(Kpt * transpose(cLinv))
end

function eval_der(tstep,mstep,xt::Array{Float64,1},yt::Array{Float64,1},
                  xprd::Array{Float64,1},yprd::Array{Float64,1},
                  mstd,R::Float64,sigR::Float64,Monotonic,Convex)
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
        for k =1:lp-1
            tmp = Phi( nu*(yprd[k]-yprd[k+1])*mstd)
            if tmp == 0.0
                der += -nupen
            else 
                der += log(tmp)
            end
        end
    end
    convcost = nu; convpen = 1.e+8
        
    if xprd[1]> 0.0 && Convex==true
        tR = abs( (yt[lt]-yprd[1])/(yt[lt-1]-yt[lt]) )

        U_idealR=R+sigR; L_idealR=R-sigR
        tmp = Phi( convcost * (U_idealR-tR) ) 
        if tmp == 0.0
            der2 += -convpen 
        else 
            der2 += log(tmp)
        end            
        if lp > 1 
            tR = abs( (yprd[1]-yprd[2])/(yt[lt]-yprd[1]) )
            Nk=Int64(xprd[2]); Nj=Int64(xprd[1])
            Nt=Int64(xt[lt]) ; Ni=Nt
            
            if abs(Nj-Nt) == 2 && abs(Nk-Nj)==2
                tmp = Phi( convcost * (U_idealR-tR)  )
            else
                tmp = 1.0
            end
            if tmp == 0.0
                der2 += -convpen
            else
                der2 += log(tmp)
            end            
            if lp>2 
                for k =1:lp-2 ###                 
                    if abs((yprd[k+1]-yprd[k+2])*mstd) < 1.e-4 ;continue;end
                    tR = abs( (yprd[k+1]-yprd[k+2])/(yprd[k]-yprd[k+1]) )
                    convcost = nu #* (0.3^k)
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
    return der, der2
end

function eval_logpost(yprd::T,mujoint::T,Sjoint::T2,tTheta::T,
                       lp::Int64,cLp::L
                       ) where {T<:Array{Float64,1},
                                T2<:Array{Float64,2},
                                L<:LowerTriangular{Float64,Array{Float64,2}}}
    if isposdef(Sjoint)
        cLinv,logdetSj=Mchole(Sjoint,lp,cLp)
        return -0.5* VtV( cLinv*(yprd-mujoint) ) - 0.5*logdetSj
    else
        return -1.e+200
    end
end

function evalPH_t(ktype,tstep::I,mstep::I,ith::I,
                  xt::T,yt::T,lt::I,muy::F,mstd::F,
                  xprd::T,yprd::T,lp::I,Theta::T,
                  Monotonic,Convex,prePH::F,mu_yt::T,mu_yp::T,
                  Rtt::T2,Rpt::T2,Rpp::T2,Ktt::T2,Kpt::T2,
                  Ktp::T2,Kpp::T2,R::F,sigR::F,
                  mujoint::T,Sjoint::T2,cLL::L,cLp::L
                  ) where {I<:Int64,F<:Float64,T<:Array{Float64,1},
                           T2<:Array{Float64,2},
                           L<:LowerTriangular{Float64,Array{Float64,2}}}
    KernelMat!(ktype,Theta,xt,lt,xprd,lp,Rtt,Rpt,Rpp,Ktt,Kpt,Kpp,false)

    try 
        cLL = cholesky(Ktt).L
    catch
        return false,0.0,-1.e+20,-1.e+20
    end
    logdetK=0.0
    @inbounds @simd for j = 1:lt
        logdetK += 2.0*log(cLL[j,j])
    end
    cLinv = inv(cLL)
    
    calcmuj!(cLinv,Kpt,Kpp,yt,mu_yt,mu_yp,lp,mujoint)
    calcSj!(cLinv,Kpt,Kpp,lp,Sjoint)
        
    logprior = 0.0 ## uniform
    
    llh  = -0.5 * VtV(cLinv*(yt-mu_yt))
    llh += -0.5*logdetK

    logpost = eval_logpost(yprd,mujoint,Sjoint,Theta,lp,cLp)

    if tstep == 1
        der,der2 = eval_der(tstep,mstep,xt,yt,xprd,yprd,mstd,R,sigR,Monotonic,Convex)
        return false,logprior,llh,logpost,der,der2
    end

    if logprior+llh+logpost-prePH >= 0
        return true,logprior,llh,logpost
    elseif log(rand()) < logprior+llh+logpost-prePH
        return true,logprior,llh,logpost
    else
        return false,logprior,llh,logpost
    end
end

function logMat52(tau::Float64,theta_r::Float64)
    tau * (1.0 + theta_r + theta_r^2 /3.0) * exp(-theta_r)
end

function logRBF(tau::Float64,theta_r::Float64)
    tau * exp(- 0.5 * theta_r^2)
end

function Mat52(tau::Float64,theta_r::Float64)
    tau * (1.0 + theta_r + theta_r^2 /3.0) * exp(-theta_r)
end

function Mat32(tau::Float64,theta_r::Float64)
    tau * (1.0 + theta_r) * exp(-theta_r)
end

function RBF(tau::Float64,theta_r::Float64)
    tau * exp(- 0.5 * theta_r^2)
end

function KernelMat!(ktype,Theta,xt::T,lt,xp::T,lp,
                    Rtt::T2,Rpt::T2,Rpp::T2,
                    Ktt::T2,Kpt::T2,Kpp::T2,
                    pder=false) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    @inbounds @simd for j=1:lt
        for i=j:lt
            tmp  = ktype(Theta[1],Theta[2]*Rtt[i,j])
            Ktt[i,j] = tmp; Ktt[j,i] = tmp 
        end
        for i=1:lp
            Kpt[i,j] = ktype(Theta[1],Theta[2]*Rpt[i,j])
        end
    end
    @inbounds @simd for j=1:lp
        for i=j:lp
            tmp  = ktype(Theta[1],Theta[2]*Rpp[i,j])
            Kpp[i,j] = tmp; Kpp[j,i] = tmp
        end
    end
    nothing
end
