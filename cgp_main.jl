using LinearAlgebra 
using Distributions
using SpecialFunctions
using Printf
using StatsBase
using Base.Threads
using LsqFit
using SIMD
using Glob
using Profile

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

function Resample(iThetas::TTA, yprds::TTA,
                  PHs::T, logp0s::T, llhs::T, logposts::T,
                  ders::TTA, mujs::TTA, SLs,SLinvs,numN::I
                  ) where {T<:Array{Float64,1},TTA<:Array{Array{Float64,1}},
                           TTA2<:Array{Array{Float64,2}},I<:Int64,F<:Float64}
    w_der=[0.0 for i=1:numN]
    x=[ [] for i=1:numN]
    Pv = [ [iThetas[ith], yprds[ith],
            PHs[ith], logp0s[ith], llhs[ith], logposts[ith],
            ders[ith], mujs[ith], SLs[ith], SLinvs[ith]] for ith=1:numN]
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
    return [ [x[ith][jj] for ith=1:numN] for jj=1:10]
end                    

function readinput(inpname,xpMax,Monotonic,Convex,ktype)
    tmp=split(inpname,"_")[(length(split(inpname,"_")))]
    Kernel=string(ktype)
    if Kernel=="logMat52" || Kernel=="logRBF"
        Tsigma = [1.0,1.0];tTheta=[5.0,5.0]
    else    
        Tsigma = [0.1,0.1];tTheta=[1.0,1.0]
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

function updateTheta!(tTheta::Array{Float64,1},sigma_T::Array{Float64,1},
                     theta0::Array{Float64,1})
    #exp.(rand(MvNormal(log.(tTheta),sigma_T)))
    theta0[1] = exp( log(tTheta[1]) + sigma_T[1] * randn() )
    theta0[2] = exp( log(tTheta[2]) + sigma_T[2] * randn() )
    nothing
end


function proposal_y!(mean::Array{Float64,1},lp::Int64,cqY::Float64,
                     cLp::LowerTriangular{Float64,Array{Float64,2}},
                     zerop::Array{Float64,1})
    try
        myMvNormal!(mean,lp,cqY,zerop,cLp,[randn() for i=1:lp])
    catch
        zerop .= mean .+ 1.e+1
    end
    nothing ##zerop is updated to c_yprd
end

function myMvNormal!(mean::Array{Float64,1},lt::Int64,cqY::Float64,
                     zerovec::Array{Float64,1},
                     cLL::LowerTriangular{Float64,Array{Float64,2}},
                     rv::Array{Float64,1})
    for i= 1:lt
        zerovec[i] = mean[i]
        for j=1:i
            zerovec[i] += cqY* cLL[i,j] * rv[j]
        end
    end
    nothing
end

function main(mstep::I,numN::I,sigRfac::F,
              ktype,inpname,inttype,
              xpMax::I,Monotonic,Convex,paramean,qT::F,qY::F,qYfac::F) where{I<:Int64,F<:Float64}
    Tsigma,tTheta,xt,yt,xprd,xun,yun,oxt,oyt,iThetas,lt,lp,muy,mstd,pfit,Rms=readinput(inpname,xpMax,Monotonic,Convex,ktype)
    Rtt,Rpt,Rpp=Rms
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
    tlogprior=-1.e+20;tllh=-1.e+20;tlogpost=-1.e+20;tder=-1.e+20;tder2=-1.e+20
    tPH =  tlogprior + tllh + tlogpost + tder + tder2
    yprds= [ 0.0*xprd .- 1.e-2 for i=1:numN]
    PHs = [ tPH for i=1:numN]; logp0s = [tlogprior for i=1:numN]
    llhs=[tllh for i=1:numN]; logposts=[tlogpost for i=1:numN]; ders=[ [tder,tder2] for i=1:numN]
  
    yt= (yt .-muy)/mstd;   ydel = yt[lt-1]-yt[lt]   
    difyt = yt-mu_yt
    
    Ktt = zeros(Float64,lt,lt); Kpp = zeros(Float64,lp,lp);Kpt = zeros(Float64,lp,lt); Ktp = zeros(Float64,lt,lp)
    Mtt = zeros(Float64,lt,lt); Mpp = zeros(Float64,lp,lp);Mpt = zeros(Float64,lp,lt); Mtp = zeros(Float64,lt,lp)
    vect = zeros(Float64,lt); vecp = zeros(Float64,lp); zerop = zeros(Float64,lp)

    muj0 = zeros(Float64,lp); Sj0 =zeros(Float64,lp,lp)
    mujs=[ muj0 for i=1:numN]
    cLL = LowerTriangular(zeros(Float64,lt,lt));cLp = LowerTriangular(zeros(Float64,lp,lp))
    cLtinv = LowerTriangular(zeros(Float64,lt,lt));cLpinv = LowerTriangular(zeros(Float64,lp,lp))    
    SLs =[ cLp for i=1:numN]; logdetSjs=[ 1.e+20 for i=1:numN]
    SLinvs =[ cLp for i=1:numN]; 
    AccT = [false]; fac0=[0.0]
    ret = [-1.e+20,-1.e+20,-1.e+20, 1.e+20 ]
    
    if paramean==true;NDist=Normal(0,0.01);else; NDist=Normal(R,5*sigR);end

    if paramean==true
        pn=sign( 0.5-rand())
        c_yprd= [ mu_yp[i] + pn*0.01*mu_yp[i] for i=1:lp ]
    else
        rtmp = rand(NDist)
        c_yprd= [ yt[lt] - ydel * rtmp * (rtmp^(i)-1.0)/(rtmp-1.0) for i=1:lp]
    end
    prePH = -1.e+150

    tstep = 1
    for ith=1:numN
        KernelMat!(ktype,iThetas[ith],xt,lt,xprd,lp,Rtt,Rpt,Rpp,Ktt,Kpt,Kpp,fac0,false)
        if paramean==true
            pn=sign( 0.5-rand())
            c_yprd= [ mu_yp[i] + pn*0.01*mu_yp[i] for i=1:lp ]
        else
            rtmp = rand(NDist)
            c_yprd= [ yt[lt] - ydel * rtmp * (rtmp^(i)-1.0)/(rtmp-1.0) for i=1:lp]
        end
        initPH!(lt,lp,c_yprd,
                iThetas[ith],prePH,difyt,mu_yp,
                Ktt,Kpt,Kpp,muj0,Sj0,
                cLL,cLp,Mtt,Mpt,Mtp,Mpp,vect,vecp,
                AccT,ret)      
        ## updated: cLL (L of Ktt) ret(logprior,llh,logpost) cLP (L of Sjoint)
        ## muj0 Sj0
        if AccT[1]
            try            
                myCholesky!(Sj0,lp,cLp) #updated cLp (L of Sj)
                ret[4]=0.0 #logdetSj
                @inbounds @simd for jj = 1:lp
                    ret[4] += 2.0*log(cLp[jj,jj])
                end
                cLpinv = inv(cLp)
                mul!(vecp,cLpinv,c_yprd-muj0)
                ret[3] = -0.5 * dot(vecp,vecp)  - 0.5 * ret[4]

                tder,tder2 = eval_der(tstep,mstep,xt,yt,
                                      xprd,c_yprd,mstd,R,sigR,Monotonic,Convex)
                tlogprior=ret[1]; tllh=ret[2]; tlogpost=ret[3]; logdetSj=ret[4]
                tPH = tlogprior+tllh+tlogpost+tder+tder2
                PHs[ith]=tPH; logp0s[ith]=tlogprior; llhs[ith]=tllh; logposts[ith]=tlogpost
                ders[ith]=[tder,tder2]
                yprds[ith]= copy(c_yprd)
                mujs[ith]=muj0
                SLs[ith] = deepcopy(cLp)
                SLinvs[ith] = deepcopy(cLpinv)
                logdetSjs[ith] = logdetSj
            catch
                nothing
            end
        end
    end
    c_Theta =[0.0, 0.0]
    sigma_T = cqT*Tsigma
    for tstep=2:mstep
        println("tstep ",tstep)
        @inbounds for ith=1:numN
            #t0 =time()  ###
            tder,tder2=ders[ith]
            tlogprior=logp0s[ith]
            tllh=llhs[ith]
            tlogpost = logposts[ith]
            #t1 = time()  ###

            ###### Accept check for Hyperparameters ###
            updateTheta!(iThetas[ith],sigma_T,c_Theta)
            #t2 = time()  ###
            KernelMat!(ktype,c_Theta,xt,lt,xprd,lp,Rtt,Rpt,Rpp,Ktt,Kpt,Kpp,fac0,false) # Ktt,Kpt,Kpp is calculated
            #t3 = time()  ###
            evalllh!(lt,lp,difyt,mu_yp,
                     Ktt,Kpt,Kpp,muj0,Sj0,
                     cLL,cLp,Mtt,Mpt,Mtp,Mpp,vect,vecp,
                     AccT,ret,cLtinv)
            ### updated: cLL(L of Ktt) muj0 Sj0<unused below> ret[1](logprior) ret[2](llh) AccT
            #t4 =time() ###
            
            if AccT[1] ### Ktt is PSD
                try
                    myCholesky!(Sj0,lp,cLp) #updated cLp (L of Sj)
                    ret[4]=0.0 #logdetSj
                    @inbounds @simd for jj = 1:lp
                        ret[4] += 2.0*log(cLp[jj,jj])
                    end
                    cLpinv .= cLp
                    LinearAlgebra.inv!(cLpinv)
                    mul!(vecp,cLpinv,yprds[ith]-muj0)
                    ret[3] = -0.5 * dot(vecp,vecp)  - 0.5 * ret[4]
                catch
                    nothing
                end
            end
            ### updated: cLp(L of Sjoint) ret[4](logdetSj)            
            
            AccT[1] = ifelse( log(rand()) <= ret[1]+ret[2]+ret[3]-tlogprior- tllh - tlogpost ,true,false)
            if AccT[1]
                iThetas[ith] = copy(c_Theta)
                tlogprior,tllh,tlogpost=ret[1],ret[2],ret[3]
                achitT += 1.
                mujs[ith]=copy(muj0)
                SLs[ith]= deepcopy(cLp)
                SLinvs[ith]=deepcopy(cLpinv)
                logdetSjs[ith] = ret[4]
            end
            #t5 = time()  ###

            ###### Accept check for yprd ###
            try
                proposal_y!(yprds[ith],lp,cqY,SLs[ith],zerop) ## updated: zerop(c_yprd)
                mul!(vecp, SLinvs[ith],zerop-mujs[ith]) ##updated: vecp
                n_logpost= -0.5*dot(vecp,vecp) -0.5 *logdetSjs[ith]
                n_der,n_der2 = eval_der(tstep,mstep,xt,yt,xprd,zerop,mstd,R,sigR,Monotonic,Convex)

                diff = n_logpost + n_der + n_der2 - tlogpost - tder - tder2
                thit = ifelse(log(rand()) <= diff, 1,0)
                if thit == 1
                    yprds[ith] = copy(zerop)
                    tlogpost = n_logpost;tder=n_der; tder2=n_der2
                    achitY += 1
                end
            catch
                nothing
            end

            #t6 = time()
            
            logp0s[ith]=tlogprior
            llhs[ith]=tllh
            logposts[ith]=tlogpost
            ders[ith]=copy([tder,tder2])
            PHs[ith] = tlogprior + tllh + tlogpost + tder +tder2

            #t7 = time()  ###
            #tsum = t7-t0 ### 
            #s1=@sprintf "%s %10.2e %s %7.1f %s %7.1f" "tsum" tsum "Read:" 100*(t1-t0)/tsum "update T:" 100*(t2-t1)/tsum 
            #s2=@sprintf "%s %7.1f %s %7.1f %s %7.1f"  "Kernel:" 100*(t3-t2)/tsum "evalllh:" 100*(t4-t3)/tsum  " Sj&logpost:" 100*(t5-t4)/tsum
            #s3=@sprintf "%s %7.1f %s %7.1f"  "yprd&logpost:" 100*(t6-t5)/tsum "write:" 100*(t7-t6)/tsum
            #println(s1,s2,s3)
        end
        #### ~Resampling~
        if  (tstep == 200 || tstep == 500 ||tstep==1000) && (Monotonic==true || Convex ==true)
            tmp = Resample(iThetas, yprds, PHs, logp0s, llhs, logposts, ders, mujs, SLs,SLinvs,numN)
            iThetas=copy(tmp[1]); yprds=deepcopy(tmp[2]); PHs=copy(tmp[3]); logp0s=copy(tmp[4]);
            llhs=copy(tmp[5]); logposts=copy(tmp[6]); ders=deepcopy(tmp[7]); mujs=deepcopy(tmp[8])
            SLs = deepcopy(tmp[9]) ; SLs = deepcopy(tmp[10]) 
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
        s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" PHs[1] "llh:" llhs[1] "logpost:" logposts[1] " der:" ders[1][1] "der2:" ders[1][2]
        println(s1,"@ith=1,Thetas=",iThetas[1])
        #if tstep %100 == 0
        #    s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100.0*achitT/(numN*(tstep-1)) " Y:" 100.0*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
        #    println(s)
        #end
        s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100.0*achitT/(numN*(tstep-1)) " Y:" 100.0*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
        println(s)
    end
    Pv=[ [ iThetas[ith], yprds[ith], 
           PHs[ith], logp0s[ith], llhs[ith], logposts[ith],
           ders[ith],mujs[ith],SLs[ith],SLinvs[ith]]  for ith=1:numN]
    E0,Evar=Summary(mstep,numN,xt,yt,xun,yun,
                    xprd,oxt,oyt,
                    Pv,muy,mstd,"Theta",inpname,inttype,
                    Monotonic,Convex,paramean)
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
        println(iot,Pv[ith][1][1]," ",Pv[ith][1][2]," ",Pv[ith][3], " ", Pv[ith][4],
                " ",Pv[ith][5]," ",Pv[ith][6], " ",Pv[ith][7])
    end
    close(iot)
    return [E0,Evar]
end

function myCholesky!(tmpA::Array{Float64,2},
                     ln::Int64,cLL::LowerTriangular{Float64,Array{Float64,2}})
    l11 = sqrt(tmpA[1,1]) 
    cLL[1,1] = l11
    cLL[2,1] = tmpA[2,1]/l11; cLL[2,2] = sqrt( tmpA[2,2]-cLL[2,1]^2)
    for i=3:ln
        for j=1:i-1
            cLL[i,j] = tmpA[i,j]
            @simd for k = 1:j-1
                cLL[i,j] += - cLL[i,k]*cLL[j,k]                
            end
            cLL[i,j] = cLL[i,j] / cLL[j,j]            
        end
        cLL[i,i] = tmpA[i,i]
        @simd for j=1:i-1
            cLL[i,i] += -cLL[i,j]^2
        end
        cLL[i,i] = sqrt(cLL[i,i])             
    end
    nothing
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

function Summary(tstep::I,numN::I,xt::T,yt::T,xun::T,yun::T,
                 xprd::T,oxt::T,oyt::T,
                 Pv,muy::F,mstd::F,plttype,inpname,inttype,
                 Monotonic::B,Convex::B,paramean::B
                 ) where{I<:Int64,F<:Float64,B<:Bool,
                         T<:Array{Float64,1}}
    global bestV,bestW, Wm,Ws,bestmuj,bestSj
    lt=length(xt); lp=length(xprd)
    yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprd2s= [ [0.0 for ith=1:numN] for kk=1:lp]
    Weights  = [0.0 for ith=1:numN]    
    WeightsH = [0.0 for ith=1:numN]
    Thetas=[[0.0 for ith=1:numN] for k=1:3]
    bestPH= -1.e+15
    bestmuj = zeros(Float64,lp); bestSj = zeros(Float64,lp,lp)
    PHs  = [0.0 for ith=1:numN]    
    for ith=1:numN
        tmp=Pv[ith]
        tTheta,typrd,tPH,tlogprior,tllh,tlogpost,tders,tmuj,tSj=tmp
        tder, tder2 = tders
        if abs(tPH - (tlogprior+tllh+tlogpost+tders[1] + tders[2])) > 1.e-8
            println("tPH -all", tPH - (tlogprior+tllh+tlogpost+tders[1] + tders[2]))
        end
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
        logw =tlogprior+tllh +tlogpost +tder + tder2 -50.0 ##to avoid overflow
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
            bestV=[tTheta,typrd,tPH,tlogprior,tllh,tlogpost,tder,tder2]
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

    
    bestT,besty=bestV[1],bestV[2]*mstd.+muy

    Oyun = yun
    cLp = LowerTriangular(zeros(Float64,lp,lp)) 

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
                  Kpt::T2,Kpp::T2,difyt::T,mu_yp::T,lp::Int64,
                  mujoint::T,vect::T,vecp::T,Mpt::T2) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    mul!(vect,cLinv,difyt)
    mul!(vect,transpose(cLinv),vect)
    mul!(vecp,Kpt,vect)
    @. mujoint = mu_yp + vecp
    return nothing
end

function calcSjMt!(cLinv::LowerTriangular{Float64,Array{Float64,2}},
                   Kpt::T2,Kpp::T2,lp::Int64,Sjoint::T2,Mtt::T2,Mpt::T2,Mtp::T2,Mpp::T2
                   ) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    transpose!(Mtt,cLinv) ## Mtt := transpose(cLinv)
    mul!(Mpt,Kpt,Mtt) ## Mpt = Kpt*transpose(cLinv)
    transpose!(Mtp,Mpt) ## Kpt:= transpose(Mpt)
    mul!(Mpp,Mpt,Mtp) ## Mpp = Mpt*transpose(Mpt)
    @inbounds @simd for j = 1:lp
        for i = 1:lp
            Sjoint[i,j] = Kpp[i,j] - Mpp[i,j] ## minus sign
        end
    end
    nothing
end


function eval_der(tstep::Int64,mstep::Int64,
                  xt::Array{Float64,1},yt::Array{Float64,1},
                  xprd::Array{Float64,1},yprd::Array{Float64,1},
                  mstd,R::Float64,sigR::Float64,Monotonic::Bool,Convex::Bool)
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

function evalllh!(lt::I,lp::I,difyt::T,mu_yp::T,
                  Ktt::T2,Kpt::T2,Kpp::T2,mujoint::T,Sjoint::T2,
                  cLL::L,cLp::L,Mtt::T2,Mpt::T2,Mtp::T2,Mpp::T2,vect::T,vecp::T,
                  AccT::BA,ret::T,cLinv::L
                  ) where {I<:Int64,F<:Float64,T<:Array{Float64,1},
                           BA<:Array{Bool,1},
                           T2<:Array{Float64,2},
                           L<:LowerTriangular{Float64,Array{Float64,2}}}
    try
        myCholesky!(Ktt,lt,cLL)
        logdetK=0.0
        @inbounds @simd for j = 1:lt
            logdetK += log(cLL[j,j])
        end
        ret[1] = 0.0 ## uniform logprior
        cLinv .= cLL
        LinearAlgebra.inv!(cLinv)
        calcmuj!(cLinv,Kpt,Kpp,difyt,mu_yp,lp,mujoint,vect,vecp,Mpt)
        calcSjMt!(cLinv,Kpt,Kpp,lp,Sjoint,Mtt,Mpt,Mtp,Mpp)
        mul!(vect,cLinv,difyt)
        ret[2] = -0.5 * dot(vect,vect) - logdetK ### llh       
        AccT[1] = true
        ### updated: cLL(L of Ktt) muj Sj ret[1](logprior) ret[2](llh)
    catch
        AccT[1] = false; ret[2] = -2.e+20
    end
    nothing 
end

function initPH!(lt::I,lp::I,yprd::T,
                 Theta::T,prePH::F,difyt::T,mu_yp::T,
                 Ktt::T2,Kpt::T2,Kpp::T2,mujoint::T,Sjoint::T2,
                 cLL::L,cLp::L,Mtt::T2,Mpt::T2,Mtp::T2,Mpp::T2,vect::T,vecp::T,
                 AccT::Array{Bool,1},ret::T
                 ) where {I<:Int64,F<:Float64,T<:Array{Float64,1},
                         T2<:Array{Float64,2},
                          L<:LowerTriangular{Float64,Array{Float64,2}}}
    try
        myCholesky!(Ktt,lt,cLL)
        logdetK=0.0
        @inbounds @simd for j = 1:lt
            logdetK += 2.0*log(cLL[j,j])
        end
        cLinv = inv(cLL)             
        calcmuj!(cLinv,Kpt,Kpp,difyt,mu_yp,lp,mujoint,vect,vecp,Mpt)
        calcSjMt!(cLinv,Kpt,Kpp,lp,Sjoint,Mtt,Mpt,Mtp,Mpp)
        ret[1] = 0.0 ## uniform logprior
        mul!(vect,cLinv,difyt)
        ret[2]  = -0.5 * dot(vect,vect) - 0.5*logdetK ### llh
        try
            myCholesky!(Sjoint,lp,cLp) # cLp:= L of Sjoint
            mul!(vecp,inv(cLp),yprd-mujoint)
            ret[3] = -0.5 * dot(vecp,vecp)  - 0.5*logdetSj ### logpost
            ret[4] = logdetSj
        catch
            ret[3] = -1.e+20
        end       
        AccT[1]=ifelse( log(rand()) < ret[1]+ret[2]+ret[3]-prePH,true,false)
    catch
        AccT[1] = false
    end
    nothing 
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
                    Ktt::T2,Kpt::T2,Kpp::T2,fac0::T,
                    pder=false) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    @inbounds for j=1:lt
        @simd for i=j:lt
            fac0[1]  = ktype(Theta[1],Theta[2]*Rtt[i,j])
            Ktt[i,j] = fac0[1]; Ktt[j,i] = fac0[1] 
        end
        for i=1:lp
            Kpt[i,j] = ktype(Theta[1],Theta[2]*Rpt[i,j])
        end
    end
    @inbounds for j=1:lp
        @simd for i=j:lp
            fac0[1]  = ktype(Theta[1],Theta[2]*Rpp[i,j])
            Kpp[i,j] = fac0[1]; Kpp[j,i] = fac0[1]
        end
    end
    nothing
end


