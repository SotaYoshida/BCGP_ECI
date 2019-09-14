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
    Rtt=zeros(Float64,lt,lt)
    Rpt=zeros(Float64,lp,lt)
    Rpp=zeros(Float64,lp,lp)
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
function Resample(Pv,numN)
    ###Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
    w_der=[0.0 for i=1:numN]
    x=[ [] for i=1:numN]
    for i =1:numN
        #tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[i]
        tmp= Pv[i][9] + Pv[i][10]
        if tmp > 709.0
            tmp= 1.e+30
        elseif tmp<-746.0
            tmp=1.0e-100
        else
            tmp=exp(tmp)
        end
        w_der[i]=tmp
    end
    #println("w_der $w_der")
    StatsBase.alias_sample!(Pv,weights(w_der),x)
    return x
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
    oxtrain,oytrain,xun,yun=make_xyt(xall,yall,useind,l1,l2,false)
    println("x(data) $oxtrain \ny(data) $oytrain")
    xtrain=oxtrain;ytrain=oytrain
    olt=length(xtrain)

    pfit=B3fit(xtrain,ytrain)
    println("B3fit $pfit")

    iThetas=[ [100*rand(),20.0*rand()] for i=1:numN]
    if Kernel=="logRBF" || Kernel=="logMatern"
        iThetas=[ [100*rand(),30.0*rand()] for i=1:numN]
    end
    if Kernel=="NSMatern" || Kernel=="Lin+Mat" || Kernel=="Lin*Mat"
        iThetas=[ [10*rand(),5.0*rand(),1.0*rand()] for i=1:numN]
    end

    lt=length(xtrain)
    Ev=0.0
    Mysigma = Diagonal( (1.e-12) * ones(Float64,lt,lt))

    #muy=mean(ytrain)
    muy=minimum(ytrain) ### (more efficient?)
    mstd=std(ytrain)

    xprd=Float64[];pNmax=[]
    unx=collect(xtrain[lt]+2:2.0:xpMax)
    for tmp in unx 
        if (tmp in xtrain)==false 
            push!(xprd,tmp)
        end
    end
    lp=length(xprd)
    Rtt,Rpt,Rpp=makeRm(xtrain,xprd,lt,lp)
    return Tsigma,tTheta,xtrain,ytrain,xprd,xun,yun,oxtrain,oytrain,iThetas,lt,lp,Mysigma,muy,mstd,pfit,Rtt,Rpt,Rpp
end
function updateyprd(typrd::Array{Float64,1},cqY::Float64,Sjoint::Array{Float64,2})            
    rand(MvNormal(typrd,cqY*Sjoint)) 
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

function main(xtrain,ytrain,yun,mstep,numN,Mysigma,muy,mstd)
    achitT=0;achitY=0
    cqT=qT;cqY=qY
    xd=[0.0]; xd2=[0.0]
    lt=length(xtrain);ld=length(xd);ld2=length(xd2);lp=length(xprd)
    yprd=0*xprd.-1.e-2; yd=zeros(Float64,ld);  yd2=zeros(Float64,ld2)
    if paramean==true
        mu_yt  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xtrain[i])  for i =1:lt],lt).-muy)/mstd
        mu_yp  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i])    for i =1:lp],lp).-muy)/mstd
    else 
        mu_yt = reshape([ 0.0 for i =1:lt],lt); mu_yp = reshape([ 0.0 for i =1:lp],lp)
    end
    meanf = vcat(mu_yt,mu_yp)
    Ktt=zeros(Float64, lt,lt); Kpp=zeros(Float64,lp,lp); Kpt = zeros(Float64,lp,lt)
    tmujoint=zeros(Float64,lp+ld+ld2);tSjoint=ones(Float64,lp+ld+ld2,lp+ld+ld2)
    Pv=[ [iThetas[i],yprd,yd,yd2,-1.e+300,-1.e+15,-1.e+15,-1.e+15,-1.e+15,-1.e+15,tmujoint,tSjoint] for i=1:numN]    
    ytrain= (ytrain .-muy)/mstd
    ydel = (ytrain[lt-1]-ytrain[lt])   
    println("xp",xprd," xt ", xtrain," yt ",ytrain," muy ",muy,"mstd",mstd, "lp $lp ld $ld ld2 $ld2 ")
    if paramean==true;NDist=Normal(0,0.001);else; NDist=Normal(R,sigR);end
    tstep = 1
    for ith=1:numN 
        tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
        tyd=0*xd; tyd2=0*xd2
        Accept,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint = evalPH_t(Kernel,tstep,mstep,ith,xtrain,ytrain,lt,Mysigma,muy,mstd,xprd,yprd,lp,xd,yd,ld,xd2,yd2,ld2,iThetas[ith],Monotonic,Convex,"Sample",false,1.e+10,meanf)
        if paramean==true
            pn=sign( 0.5-rand())
            c_yprd= [ mu_yp[i] + pn*0.005*mu_yp[i] for i=1:lp ]
        else
            rtmp = rand(NDist);c_yprd= [ ytrain[lt] - ydel * rtmp * (rtmp^(i)-1)/(rtmp-1) for i=1:lp]
        end
        tder,tder2 = eval_der(tstep,mstep,xtrain,ytrain,xprd,c_yprd,mstd,R,sigR)
        tPH = tlogprior+tllh+tlogpost+tder+tder2
        Pv[ith]=[iThetas[ith],c_yprd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
    end    
    sigma_T = cqT*Tsigma
    c_yd=0*xd;c_yd2=xd2*0   #### to be changed if derKer introduced         
    for tstep=2:mstep
        println("tstep ",tstep)
        @time @threads for ith=1:numN 
        #@time for ith=1:numN 
            #t0 =time()

            #tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith][1:
            tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2=Pv[ith][1:10]
            prePH = tlogprior + tllh + tlogpost
            
            #t1 = time()

            c_Theta = updateTheta(tTheta,sigma_T)

            #t2 = time()
 
            tmp=evalPH_t(Kernel,tstep,mstep,ith,xtrain,ytrain,lt,
                         Mysigma,muy,mstd,xprd,typrd,lp,xd,tyd,ld,
                         xd2,tyd2,ld2,c_Theta,
                         Monotonic,Convex,"Sample",false,prePH,meanf)
            
            #t3 = time()

            if tmp[1]==true 
                tTheta = c_Theta
                tlogprior,tllh,tlogpost,tmujoint,tSjoint=tmp[2],tmp[3],tmp[4],tmp[5],tmp[6]
                tPH = tlogprior + tllh + tlogpost + tder + tder2
                achitT += 1
            else 
                tmujoint=Pv[ith][11];tSjoint=Pv[ith][12]
            end

            if ith == 1 
                s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" tPH "   llh:" tllh "   logpost:" tlogpost "   der:" tder "der2:" tder2
                println(s1,"@",tTheta)
            end

            #t4 = time()

            c_yprd = updateyprd(typrd,cqY,tSjoint)
            #c_yprd = updateyprd_m(ytrain[lt],ydel,lp,R,sigR)
            #c_yprd = updateyprd_n(typrd,lp,mstd)           

            #t5 = time()

            n_logpost=eval_logpost(c_yprd,tmujoint,tSjoint)

            #t6 = time()

            n_der,n_der2 = eval_der(tstep,mstep,xtrain,ytrain,xprd,c_yprd,mstd,R,sigR)

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
                typrd=c_yprd;tyd=c_yd; achitY += 1
                tPH = tlogprior + tllh + tlogpost + tder + tder2 
            end          
            #t8 = time()
            if tmp[1]==true || yAccept==true
                Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
            end
            # t9 = time()
            # tsum = t9-t0
            # s1=@sprintf "%s %10.2e %s %8.1f %s %8.1f %s %8.1f %s %8.1f" "tsum" tsum "t1-0:" 100*(t1-t0)/tsum "t2-1:" 100*(t2-t1)/tsum "t3-2:" 100*(t3-t2)/tsum "t4-t3:" 100*(t4-t3)/tsum
            # s2=@sprintf "%s %8.1f %s %8.1f" " t5-4:" 100*(t5-t4)/tsum "t6-5:" 100*(t6-t5)/tsum 
            # s3=@sprintf "%s %8.1f %s %8.1f %s %8.1f" " t7-6:" 100*(t7-t6)/tsum "t8-7:" 100*(t8-t7)/tsum "t9-8:" 100*(t9-t8)/tsum
            # println(s1,s2,s3)
            if ith==1
                s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" newPH+tllh+tlogprior "   llh:" tllh "   logpost:" n_logpost "   der:" n_der "der2:" n_der2
                println(s1,"@",tTheta)
                #println("c_yprd ",c_yprd*mstd.+muy)
                #ratio=calc_ratio(c_yprd,ytrain[lt-1],ytrain[lt],mstd,muy)
                #println("y ratio ",ratio)
                #println("tmujoint[1:lp]", tmujoint[1:lp] *mstd .+ muy)
            end

        end
        ### ~Resampling~
        if  tstep == 200 || tstep == 1000
            Pv=Resample(Pv,numN)
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
    E0,Evar=Summary(mstep,numN,xtrain,ytrain,xun,yun,xprd,xd,Pv,muy,mstd,"Theta")
    Nmin=string(Int64(oxtrain[1])) ;  Nmax=string(Int64(oxtrain[length(oxtrain)]))
    iot = open("Thetas_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*".dat", "w")
    for ith = 1:numN
        println(iot,Pv[ith][1][1]," ",Pv[ith][1][2]," ",Pv[ith][5], " ", Pv[ith][6]," ",Pv[ith][7] )
    end
    close(iot)

    return [E0,Evar]
end

function detR(xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)
    #global achitT,achitY#,Tsigma,Pv
    #global Wm, Ws
    achitT=0;achitY=0
    cqT=qT;cqY=qY
    tnumN=numN
    lt=length(xtrain)
    xprd=[xtrain[lt]+2]
    yprd=0*xprd
    lp=length(xprd)
    
    xd=[0.0]; xd2=[0.0]
    ld=length(xd);ld2=length(xd2)    
    yd=zeros(Float64,ld)
    yd2=zeros(Float64,ld2)
    
    tmujoint=zeros(Float64,lp+ld+ld2);tSjoint=ones(Float64,lp+ld+ld2,lp+ld+ld2)
    mu_yt = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xtrain[i]) for i =1:lt],lt).-muy)/mstd
    mu_yp  = (reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i]) for i =1:lp],lp).-muy)/mstd
    meanf = vcat(mu_yt,mu_yp)
    iThetas=[ [100*rand(),30.0*rand()] for i=1:tnumN]
    Pv=[ [iThetas[i],yprd,yd,yd2,-1.e+300,-1.e+5,-1.e+5,-1.e+5,-1.e+5,-1.e+5,tmujoint,tSjoint] for i=1:tnumN]
    R=0.5; sigR=0.3
    NDist=Normal(R,sigR)
    ytrain= (ytrain .-muy)/mstd
    ydel = (ytrain[lt-1]-ytrain[lt])   
    Ktt=zeros(Float64, lt,lt); Kpp=zeros(Float64,lp,lp); Kpt = zeros(Float64,lp,lt)
    for tstep=1:Rstep
        #@threads for ith=1:tnumN 
        for ith=1:tnumN 
            tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
            tder2 = 0.0
            if tstep == 1                
                tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
                tyd=0*xd; tyd2=0*xd2
                Accept,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint = evalPH_t(Kernel,tstep,mstep,ith,xtrain,ytrain,lt,Mysigma,muy,mstd,xprd,yprd,lp,xd,yd,ld,xd2,yd2,ld2,iThetas[ith],Monotonic,Convex,"Sample",false,1.e+10,meanf)
                rtmp = rand(NDist)                
                c_tmp= [ ytrain[lt] - ydel * rtmp * (rtmp^(i)-1)/(rtmp-1) for i=1:length(tmujoint)];  c_yprd=c_tmp[1:lp]                
                tPH = -1.e+15
                Pv[ith]=[iThetas[ith],c_yprd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]          
            else 
                prePH = tlogprior + tllh + tlogpost 
                c_Theta = updateTheta(tTheta,cqT*Tsigma)
                tmp=evalPH_t(Kernel,tstep,mstep,ith,xtrain,ytrain,lt,Mysigma,muy,mstd,xprd,typrd,lp,xd,tyd,ld,xd2,tyd2,ld2,c_Theta,Monotonic,Convex,"Sample",false,prePH,meanf)
                if tmp[1]==true 
                    tTheta = c_Theta
                    tlogprior,tllh,tlogpost,tmujoint,tSjoint=tmp[2],tmp[3],tmp[4],tmp[5],tmp[6]
                    tPH = tlogprior + tllh + tlogpost + tder + tder2
                    achitT += 1
                end
                yjoint = typrd
                c_yprd = updateyprd(typrd,cqY,tSjoint)
                c_yd=0*xd;c_yd2=xd2*0
                n_logpost=eval_logpost(c_yprd,tmujoint,tSjoint)
                n_der,n_der2 = eval_der(tstep,mstep,xtrain,ytrain,xprd,c_yprd,mstd,R,sigR)
                n_der2 = 0.0
                prePH = tlogpost  + tder  + tder2
                newPH = n_logpost + n_der + n_der2
                Accept=false
                if newPH - prePH >= 0.0
                    Accept=true
                elseif log(rand()) < newPH-prePH
                    Accept=true
                end
                if Accept==true 
                    tlogpost=n_logpost; tder = n_der; tder2 = 0.0
                    typrd=c_yprd;tyd=c_yd
                    achitY += 1
                    tPH = tlogprior + tllh + tlogpost + tder + tder2 
                end          
                Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,0.0,tmujoint,tSjoint] ## tder2==0.0
                #if ith == 1
                #    s1 = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" newPH+tllh+tlogprior "   llh:" tllh "   logpost:" n_logpost "   der:" n_der "der2:" n_der2
                #    println(s1, " c_yprd ",c_yprd[1]*mstd+muy," @ $tTheta")
                #end
            end
        end
        if tstep % 3 == 0 && 100 <= tstep <= 1000
            Pv=Resample(Pv,tnumN)
        end
        if 1000 > tstep > 100
            lqT=log(cqT);lqY=log(cqY)
            b_n = bn(tstep-100)
            lqT = lqT + b_n * (achitT/(tnumN*tstep) - rMH[1]) 
            lqY = lqY + b_n * (achitY/(tnumN*tstep) - rMH[2]) 
            cqT = exp(lqT)
            cqY = exp(lqY)
        end
        if tstep == Rstep
            s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Acchit T:" 100*achitT/(tnumN*(Rstep-1)) " Y:" 100*achitY/(tnumN*(Rstep-1)) " qT,qY= " cqT cqY
            println(s)
            Wm,Ws,yp=Summary(tstep,tnumN,xtrain,ytrain,xun,yun,xprd,xd,Pv,muy,mstd,"R")       
        end    
    end    
    return Wm,Ws
end


function Mchole(tmpA::Array{Float64,2},ln::Int64) 
    #cL=cholesky(tmpA, Val(false); check = true)
    cLL=cholesky(tmpA, Val(false); check = true).L
    logLii=0.0
    for i = 1:ln
        logLii += log(cLL[i,i])
    end
    return inv(cLL), 2.0*logLii
end

function Simple_inv(tmpA::Array{Float64,2},ln) 
    #println("logdet(tmpA) ",logdet(tmpA)," logLii $logLii")
    try
       return  inv(tmpA),logdet(tmpA)
    catch
        cLL=cholesky(tmpA, Val(false); check = true).L
        logLii=0.0
        for i = 1:ln
            logLii += log(cLL[i,i])
        end
        return inv(tmpA),2.0*logLii
    end
end

function nu_t(tstep,maxstep) 
    #fac = 1.e+1 # fac = 1.e+2
    fac = 1.e+6
    return fac * (tstep/maxstep)^0.1
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

function Summary(tstep,numN,xt,yt,xun,yun,xprd,xd,Pv,muy,mstd,plttype)   
    global bestPH,bestV,bestW, Wm, Ws
    lt=length(xt); lp=length(xprd); ld=length(xd)
    yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprd2s= [ [0.0 for ith=1:numN] for kk=1:lp]
    yds= [ [0.0 for ith=1:numN] for kk=1:ld]
    Weights  = [0.0 for ith=1:numN]    
    WeightsH = [0.0 for ith=1:numN]    
    iniThetas=[[0.0 for ith=1:numN] for k=1:3]      
    Thetas=[[0.0 for ith=1:numN] for k=1:3]
    for ith=1:numN
        tmp=Pv[ith]
        tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=tmp
        if length(tTheta)==2
            for kk=1:2 
                Thetas[kk][ith] = tTheta[kk]
                iniThetas[kk][ith]=iThetas[ith][kk]
            end
        else 
            for kk=1:3
                Thetas[kk][ith] = tTheta[kk]
                iniThetas[kk][ith]=iThetas[ith][kk]
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
        Nmin=string(Int64(oxtrain[1])) 
        Nmax=string(Int64(oxtrain[length(oxtrain)]))
        fn="Posterior_"*string(inttype)*"_paramean_"*string(paramean)*"_min"*Nmin*"max"*Nmax*".dat"
        io = open(fn, "w")
        println(io,inpname)
        println(io,"xprd=")
        println(io,xprd)
        println(io,"\n means=")
        println(io,means)
        println(io,"\n stds=")
        println(io,stds)
        println(io,"#weights are in logscale")
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

# function calcSj(cLinv::LowerTriangular{Float64,Array{Float64,2}},Kpt::Array{Float64,2},Kpp::Array{Float64,2},yt::Array{Float64,1},mu_yt::Array{Float64,1},mu_yp::Array{Float64,1})
#     tKtp=cLinv*transpose(Kpt)
#     mu_yp + Kpt*(transpose(cLinv)*(cLinv*(yt-mu_yt))) , Kpp - transpose(tKtp)*tKtp
# end

function calcSj(cLinv::LowerTriangular{Float64,Array{Float64,2}},Kpt::Array{Float64,2},Kpp::Array{Float64,2},yt::Array{Float64,1},mu_yt::Array{Float64,1},mu_yp::Array{Float64,1})
    mul!(tKtp,cLinv,transpose(Kpt))
    mu_yp + Kpt*(transpose(cLinv)*(cLinv*(yt-mu_yt))) , Kpp - transpose(tKtp)*tKtp
end

function eval_der_sigma(tstep,mstep,yprd::Array{Float64,1},yd::Array{Float64,1},yd2::Array{Float64,1},Monotonic,Convex)
    ### for derivative obs.
    der = 0.0; der2 = 0.0
    nu  = nu_t(tstep,mstep)
    for kk=1:length(yd)
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
        for k =1:lp-1
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
    Sjinv,logdetSj = Simple_inv(Sjoint,length(mujoint)) 
    yjoint = yprd
    term1 = -0.5 * transpose(yjoint-mujoint)*(Sjinv*(yjoint-mujoint))
    if term1 > 0.0 ## This may occur if sigma is "unphysical" (breaking PSD)
        term1 = term1 - 1.e+15
        println("warn:negative term1")
    end
    return term1  - 0.5 * logdetSj
end



function evalPH_t(Kernel,tstep,mstep,ith,xt::T,yt::T,lt,
                  Mysigma,muy,mstd,xprd::T,yprd::T,lp,xd::T,yd::T,ld,
                  xd2::T,yd2::T,ld2,Theta,
                  Monotonic,Convex,GPmode,fixR,prePH,meanf) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    #time0 = time()

    if Kernelder == true ## Not complete
        KernelMat!(Ktt,Kernel,Theta,xt,xt,0,false) + Diagonal( (1.e-8) * ones(Float64,lt,lt))
        KernelMat!(Kpp,Kernel,Theta,xprd,xprd,0,false) + Diagonal( (1.e-8) * ones(Float64,lp,lp))
        KernelMat!(Kpt,Kernel,Theta,xprd,xt,0,false) 
        KernelMat!(Kdd,Kernel,Theta,xd,xd,2,false) + Diagonal( (1.e-8) * ones(Float64,ld,ld))
        KernelMat!(Kpd,Kernel,Theta,xprd,xd,1,false) 
        KernelMat!(Ktd,Kernel,Theta,xt,xd,1,false) 
        cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))        
        if Convex==true 
            K02=KernelMat!(Kernel,Theta,xt,xd2,3,false) 
            Kp2=KernelMat!(Kernel,Theta,xprd,xd2,3,false) 
            K12=KernelMat!(Kernel,Theta,xd,xd2,4,false) 
            ld2=length(xd2)
            K22=KernelMat!(Kernel,Theta,xd2,xd2,5,false) + Diagonal( (1.e-4) * ones(Float64,ld2,ld2))
            A0 = hcat(hcat(Kpp,Kpd),Kp2)
            A1 = hcat(hcat(transpose(Kpd),Kdd),K12)
            A2 = hcat(hcat(transpose(Kp2),transpose(K12)),K22)
            A  = vcat(vcat(A0,A1),A2)
            C  = vcat(vcat(Kpt,transpose(Ktd)),transpose(K02)) 
        else
            A0 = hcat(Kpp,Kpd)
            A1 = hcat(transpose(Kpd),Kdd)
            A  = vcat(A0,A1)
            C  = vcat(Kpt,transpose(Ktd))
        end

        mu_y = reshape([ 0.0 for i=1:lp],lp)
        mu_y = reshape([pfit[1] + pfit[2]*exp(-pfit[3]*xprd[i]) for i =1:lp],lp)
        mu_yt= reshape([ 0.0 for i=1:lt],lt)
        mu_d = reshape([ 0.0 for i=1:ld],ld)
        if Convex==true
            mu_d2= reshape([ 0.0 for i=1:length(xd2)],length(xd2))
            mujoint = vcat(vcat(mu_y,mu_d),mu_d2)
        else
            mujoint = vcat(mu_y,mu_d)
        end
        mujoint += C*Kttinv*yt        
        tmp=cLinv*transpose(C)
        Sjoint = A - transpose(tmp)*tmp
        errc = 0
        while isposdef(Sjoint)==false
            errc += 1
            ttmp=eigvals(Sjoint)
            minEV=minimum(ttmp)
            if minEV==0.0
                minEV=1.e-15
            end
            Sjoint += Diagonal( 10*abs(minEV) * ones(Float64,length(mujoint),length(mujoint)))
            if errc > 100;println("NotPSD:Sjoint: eigvals",eigvals(Sjoint), " @ ",Theta, "isposdef(Sjoint)", isposdef(Sjoint) );end
        end
    else
        #t0_0 = time()

        Ktt=KernelMat(Kernel,Theta,xt,xt,lt,lt,0,Rtt,false,false)  + Mysigma
        
        #t0_1 = time()
        
        Kpp=KernelMat(Kernel,Theta,xprd,xprd,lp,lp,0,Rpp,false,false)
        
        #t0_2 = time()
        
        Kpt=KernelMat(Kernel,Theta,xprd,xt,lp,lt,0,Rpt,false,false) 
        
        if isposdef(Ktt) == false;println("issymmetric(Ktt)",issymmetric(Ktt), " @Theta=", Theta);println("eigvals:",eigvals(Ktt));end

        #t0_3 = time()
        
        cLinv,logdetK = Mchole(Ktt,lt)
        #Kttinv,logdetK = Simple_inv(Ktt)        
        
        #t0_4 = time()
        mu_yt = meanf[1:lt]
        mu_yp = meanf[lt+1:lt+lp]
        mujoint,Sjoint  = calcSj(cLinv,Kpt,Kpp,yt,mu_yt,mu_yp)
        #t0_5 = time()

        errc = 0
        while isposdef(Sjoint)==false
            ttmp=eigvals(Sjoint)
            minEV=ttmp[1]              
            Sjoint = Sjoint + Diagonal( 100*abs(minEV) * ones(Float64,lp,lp))
            errc += 1
            if errc > 100;println("NotPSD:Sjoint L516:@",Theta,ttmp);end
        end 
        #t0_6 = time()
        #tsum= t0_6-t0_0
        #s1=@sprintf "Time for KernelMat  %s %10.2e %s %8.1f %s %8.1f" "tsum" tsum "t1-0:" 100*(t0_1-t0_0)/tsum "t2-1:" 100*(t0_2-t0_1)/tsum 
        #s2=@sprintf " %s %8.1f %s %8.1f" "t3-2:" 100*(t0_3-t0_2)/tsum "t4-3:" 100*(t0_4-t0_3)/tsum 
        #s3=@sprintf " %s %8.1f %s %8.1f" "t5-4:" 100*(t0_5-t0_4)/tsum "t6-5:" 100*(t0_6-t0_5)/tsum
        #println(s1,s2,s3,"\n")
    end

    #time1 = time()

    logprior = 0.0 ## uniform     
    term1 = -0.5*transpose(yt-mu_yt)*(transpose(cLinv)*(cLinv*(yt-mu_yt)))
    term2 = -0.5*logdetK
    llh = term1 + term2 

    #time2 = time()

    Accept =false
    logpost= eval_logpost(yprd,mujoint,Sjoint)
    if tstep == 1
        ###########
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


function KernelMat(ktype,Theta,x::T,y::T,lenx,leny,kind,Rm::T2,pder=false,diag=false) where {T<:Array{Float64,1},T2<:Array{Float64,2}}
    K = zeros(Float64,lenx,leny)
    if ktype == "Matern" ## Matern Kernel with nu = 5/2
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(5)/sigma
        if pder==false 
            #K = zeros(Float64, length(x), length(y))
            for j =1:length(x)
                for k =1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    elseif kind==1    
                        f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                    elseif kind==2
                        f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    end
                    K[j,k]=f
                end
            end    
        else
            dKds = zeros(Float64, length(x), length(y))
            d2Kdsds = zeros(Float64, length(x), length(y))
            if length(x)!=length(y);println("err in KernalMat");end
            for j =1:length(x)
                for k =1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty))                             
                        # elseif kind==1    
                        #     f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                        # elseif kind==2
                        #     f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    end
                    K[j,k]=f
                    r=abs(tx-ty)
                    dKds[j,k]= f * 5*r^2 * (sigma+sqrt(5)*r) / ( 3*sigma^4 + 3*sqrt(5)*r*sigma^3 +5*r^2  )
                    d2Kdsds[j,k]= f * 5*r^2 * (5*r^2 -sigma^2 -sqrt(5)*r*sigma)/( (5*r^2+3*sqrt(5)*r*sigma +3*sigma^2)  * sigma^4 ) 
                end
            end    
            G=K + Diagonal( (1.e-8) * ones(Float64,lt,lt))
            cLinv,Ginv,logdetK = Mchole(G,length(x))                
            Gamma=Ginv*ytrain
            dKdt = K/tau
            d2Kdsdt = dKds/tau
            # d2Hdt2  = -0.5/(tau*tau) * transpose(ytrain) * Kinv * ytrain
            # d2Hdtds = -0.5/tau * transpose(Gamma)*dKds*Gamma
            # d2Hds2  =  0.5 * transpose(Gamma) * Elds2 * Gamma
            term1 = -1.0/(tau*tau) * transpose(Gamma) * K * Ginv * K * Gamma
            term2 = 0.5/(tau*tau) * tr( Ginv*K*Ginv*K)
            d2Hdt2  = term1 + term2
            term1 = -1.0/tau * transpose(Gamma) * dKds * Ginv * K * Gamma 
            term2 =  0.5/tau * tr( Ginv*dKds*Ginv*K)
            term3 = 0.5/tau * transpose(Gamma)*dKds*Gamma
            term4 = -0.5/tau * tr(Ginv*dKds)
            d2Hdtds = term1 + term2 + term3 + term4                
            term1 = -transpose(Gamma) *dKds * Ginv*dKds*Gamma
            term2 = 0.5 * tr( Ginv*dKds*Ginv*dKds)
            term3 = 0.5 * transpose(Gamma)*d2Kdsds*Gamma
            term4 = -0.5*tr(Ginv*d2Kdsds)
            d2Hds2 = term1 + term2 + term3 + term4
            return K,d2Hdt2,d2Hds2,d2Hdtds
        end
    elseif ktype == "Mat32" ## Matern Kernel with nu = 5/2
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(3)/sigma
        if pder==false 
            for j =1:length(x)
                for k =1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f=tau * ( 1 + theta*abs(tx-ty) ) * exp(-theta*abs(tx-ty)) 
                    elseif kind==1    
                        f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                    elseif kind==2
                        f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    end
                    K[j,k]=f
                end
            end    
            return K
        end
    elseif ktype == "logMat32"
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(3)/sigma
        ep = 0.0
        if pder==false
            if x==y ; k1 = j; else ;k1 = 1; end
            if diag == true ; k2 = j;else;k2 = length(y); end
            for j =1:length(x)
                for k =k1:k2
                    lx=log(x[j]+ep);ly=log(y[k]+ep)
                    if x[j]==y[k]
                        r = 0.0
                        efac=1.0
                    else
                        r = lx-ly
                        efac=exp(-theta*abs(r))
                    end
                    if kind==0
                        f=tau * ( 1.0 + theta*abs(r) ) * efac 
                    elseif kind==1    
                        f = -tau* theta^2 / 3.0 * sign(ty-tx) * efac * (abs(r)* theta * abs(r)^2)  / (ty+ep) ##for 5/2
                    elseif kind==2
                        f=tau/3.0 * theta * theta * (1 +theta*abs(r)- theta^2 * r^2) * efac / ((tx+ep)*(ty+ep)) ### for 5/2
                    end
                    K[j,k]=f
                    if x==y && j < k 
                        K[k,j]=f
                    end
                end
            end    
        else
            exit()
        end
    elseif ktype == "logMatern"
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(5)/sigma; theta2= theta^2
        #ep = 1.e-30; ep = 0.0
        if x==y ; xysame=true; else; xysame=false;end
        if pder==false
            if xysame == true
                @inbounds for j=1:lenx
                    K[j,j] = tau
                end
                @inbounds for j =1:lenx
                    @inbounds for k =j+1:leny
                        r = Rm[j,k]
                        f= (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * tau * exp(-theta*r)
                        K[j,k]=f
                        K[k,j]=f
                    end
                end
            else
                @inbounds for j =1:lenx
                    @inbounds for k =1:leny
                        r = Rm[j,k]
                        f= (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * tau * exp(-theta*r)
                        K[j,k]=f
                    end
                end        
            end    
            
            # @inbounds for j =1:lenx
            #     if xysame ; k1 = j; else ;k1 = 1; end
            #     if diag == true ; k2 = j;else;k2 = leny; end
            #     @inbounds for k =k1:k2
            #         lx=log(x[j]);ly=log(y[k])
            #         if x[j]==y[k]
            #             K[j,k]=tau
            #             continue
            #         else
            #             r = abs(lx-ly)
            #             efac=tau*exp(-theta*r)
            #             K[j,k] = (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * efac
            #             if xysame && j < k 
            #                 K[k,j] = (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * efac
            #             end
            #         end
            #     end
            # end    
        else
            exit()
        end
    elseif ktype=="RBF" ##RBF Kernel ## parameter derivative is not implemeted
        tau,sigma=Theta[1],Theta[2]
        theta=1.0/sigma
        if pder==false 
            K = zeros(Float64, length(x), length(y))
            for j =1:length(x)
                if x==y 
                    k1 = j
                else 
                    k1 = 1
                end
                for k =k1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f=tau *  exp(-0.5 * (tx-ty)^2 * theta^2 )
                    elseif kind==1    
                        f=tau * (tx-ty) * theta^2 *  exp(-0.5* (tx-ty)^2 * theta^2 )
                    elseif kind==2
                        f=tau * theta^2 * ( 1 - theta^2 * (ty-tx)^2  )  *  exp(-0.5 * (tx-ty)^2 * theta^2 )
                    elseif kind==3 
                        f= tau * ( theta^4 *  (tx-ty)^2 - theta^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
                    elseif kind==4
                        f= tau * (tx-ty) * (3 * theta^4 - theta^6 *(tx-ty)^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
                    elseif kind==5
                        f= tau * theta^4 * (3- 6 * theta^2 * (tx-ty)^2 + theta^4 * (tx-ty)^4) *  exp(-0.5 * theta^2 * (tx-ty)^2 )
                    end
                    K[j,k]=f
                    if x==y && j < k 
                        K[k,j]=f
                    end
                end
            end    
            return K
        else
            println("Deribative w.r.t. hyperparameter is not implemented for RBF Kernel!!");exit()
        end
    elseif ktype=="logRBF" ##RBF Kernel ## parameter derivative is not implemeted
        tau,sigma=Theta[1],Theta[2]
        theta=1.0/(sigma*sigma)
        ep = 1.e-30
        ep = 0.0
        if pder==false 
            for j =1:length(x)
                if x==y 
                    k1 = j
                else 
                    k1 = 1
                end
                if diag == true 
                    k2 = j
                else
                    k2 = length(y)
                end
                for k =k1:k2
                    tx=x[j];ty=y[k]
                    lx=log(tx+ep);ly=log(ty+ep)                    
                    r = lx-ly
                    exfac=exp(-0.5 * abs(r)^2 * theta )
                    if x[j]==y[k]
                        r = 0.0
                        exfac= 1.0
                    end
                    if kind==0
                        f= tau * exfac
                    elseif kind==1    
                        f= tau * theta / ty * r * exfac
                    elseif kind==2
                        f= tau * theta *( 1-theta* r^2) /(tx*ty) * exfac
                    elseif kind==3 
                        f= tau * theta / (ty*ty) * (theta * r^2 - r -1) * exfac
                    elseif kind==4
                        f= tau * theta / (tx*ty*ty) *( -theta^2 * r^3 + theta * r^2 + 3*theta*r -1) * exfac
                    elseif kind==5
                        f= tau * theta / (tx*tx*ty*ty) * (theta^3 * r^4 - theta*(1+6*theta)*r^2 + 1 + 3*theta) *exfac
                    end
                    K[j,k]=f
                    if x==y && j < k 
                        K[k,j]=f
                    end
                end
            end    
            return K
        else
            println("Deribative w.r.t. hyperparameter is not implemented for RBF Kernel!!");exit()
        end
    end
    return K
end



# function KernelMat(ktype,Theta,x::T,y::T,lenx,leny,kind,pder=false,diag=false) where {T<:Array{Float64,1}}
#     K = zeros(Float64,lenx,leny)
#     if ktype == "Matern" ## Matern Kernel with nu = 5/2
#         tau,sigma=Theta[1],Theta[2]
#         theta = sqrt(5)/sigma
#         if pder==false 
#             #K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 for k =1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     elseif kind==1    
#                         f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                     elseif kind==2
#                         f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     end
#                     K[j,k]=f
#                 end
#             end    
#         else
#             dKds = zeros(Float64, length(x), length(y))
#             d2Kdsds = zeros(Float64, length(x), length(y))
#             if length(x)!=length(y);println("err in KernalMat");end
#             for j =1:length(x)
#                 for k =1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty))                             
#                         # elseif kind==1    
#                         #     f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                         # elseif kind==2
#                         #     f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     end
#                     K[j,k]=f
#                     r=abs(tx-ty)
#                     dKds[j,k]= f * 5*r^2 * (sigma+sqrt(5)*r) / ( 3*sigma^4 + 3*sqrt(5)*r*sigma^3 +5*r^2  )
#                     d2Kdsds[j,k]= f * 5*r^2 * (5*r^2 -sigma^2 -sqrt(5)*r*sigma)/( (5*r^2+3*sqrt(5)*r*sigma +3*sigma^2)  * sigma^4 ) 
#                 end
#             end    
#             G=K + Diagonal( (1.e-8) * ones(Float64,lt,lt))
#             cLinv,Ginv,logdetK = Mchole(G,length(x))                
#             Gamma=Ginv*ytrain
#             dKdt = K/tau
#             d2Kdsdt = dKds/tau
#             # d2Hdt2  = -0.5/(tau*tau) * transpose(ytrain) * Kinv * ytrain
#             # d2Hdtds = -0.5/tau * transpose(Gamma)*dKds*Gamma
#             # d2Hds2  =  0.5 * transpose(Gamma) * Elds2 * Gamma
#             term1 = -1.0/(tau*tau) * transpose(Gamma) * K * Ginv * K * Gamma
#             term2 = 0.5/(tau*tau) * tr( Ginv*K*Ginv*K)
#             d2Hdt2  = term1 + term2
#             term1 = -1.0/tau * transpose(Gamma) * dKds * Ginv * K * Gamma 
#             term2 =  0.5/tau * tr( Ginv*dKds*Ginv*K)
#             term3 = 0.5/tau * transpose(Gamma)*dKds*Gamma
#             term4 = -0.5/tau * tr(Ginv*dKds)
#             d2Hdtds = term1 + term2 + term3 + term4                
#             term1 = -transpose(Gamma) *dKds * Ginv*dKds*Gamma
#             term2 = 0.5 * tr( Ginv*dKds*Ginv*dKds)
#             term3 = 0.5 * transpose(Gamma)*d2Kdsds*Gamma
#             term4 = -0.5*tr(Ginv*d2Kdsds)
#             d2Hds2 = term1 + term2 + term3 + term4
#             return K,d2Hdt2,d2Hds2,d2Hdtds
#         end
#     elseif ktype == "Mat32" ## Matern Kernel with nu = 5/2
#         tau,sigma=Theta[1],Theta[2]
#         theta = sqrt(3)/sigma
#         if pder==false 
#             for j =1:length(x)
#                 for k =1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=tau * ( 1 + theta*abs(tx-ty) ) * exp(-theta*abs(tx-ty)) 
#                     elseif kind==1    
#                         f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                     elseif kind==2
#                         f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     end
#                     K[j,k]=f
#                 end
#             end    
#             return K
#         end
#     elseif ktype == "logMat32"
#         tau,sigma=Theta[1],Theta[2]
#         theta = sqrt(3)/sigma
#         ep = 0.0
#         if pder==false
#             if x==y ; k1 = j; else ;k1 = 1; end
#             if diag == true ; k2 = j;else;k2 = length(y); end
#             for j =1:length(x)
#                 for k =k1:k2
#                     lx=log(x[j]+ep);ly=log(y[k]+ep)
#                     if x[j]==y[k]
#                         r = 0.0
#                         efac=1.0
#                     else
#                         r = lx-ly
#                         efac=exp(-theta*abs(r))
#                     end
#                     if kind==0
#                         f=tau * ( 1.0 + theta*abs(r) ) * efac 
#                     elseif kind==1    
#                         f = -tau* theta^2 / 3.0 * sign(ty-tx) * efac * (abs(r)* theta * abs(r)^2)  / (ty+ep) ##for 5/2
#                     elseif kind==2
#                         f=tau/3.0 * theta * theta * (1 +theta*abs(r)- theta^2 * r^2) * efac / ((tx+ep)*(ty+ep)) ### for 5/2
#                     end
#                     K[j,k]=f
#                     if x==y && j < k 
#                         K[k,j]=f
#                     end
#                 end
#             end    
#         else
#             exit()
#         end
#     elseif ktype == "logMatern"
#         tau,sigma=Theta[1],Theta[2]
#         theta = sqrt(5)/sigma; theta2= theta^2
#         #ep = 1.e-30; ep = 0.0
#         if x==y ; xysame=true; else; xysame=false;end
#         if pder==false
#             if xysame == true
#                 @inbounds for j=1:lenx
#                     K[j,j] = tau
#                 end
#                 @inbounds for j =1:lenx
#                     @inbounds for k =j+1:leny
#                         lx=log(x[j]);ly=log(y[k])
#                         r = abs(lx-ly)
#                         f= (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * tau * exp(-theta*r)
#                         K[j,k]=f
#                         K[k,j]=f
#                     end
#                 end
#             else
#                 @inbounds for j =1:lenx
#                     @inbounds for k =1:leny
#                         lx=log(x[j]);ly=log(y[k])
#                         r = abs(lx-ly)
#                         f= (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * tau * exp(-theta*r)
#                         K[j,k]=f
#                     end
#                 end        
#             end    
            
#             # @inbounds for j =1:lenx
#             #     if xysame ; k1 = j; else ;k1 = 1; end
#             #     if diag == true ; k2 = j;else;k2 = leny; end
#             #     @inbounds for k =k1:k2
#             #         lx=log(x[j]);ly=log(y[k])
#             #         if x[j]==y[k]
#             #             K[j,k]=tau
#             #             continue
#             #         else
#             #             r = abs(lx-ly)
#             #             efac=tau*exp(-theta*r)
#             #             K[j,k] = (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * efac
#             #             if xysame && j < k 
#             #                 K[k,j] = (1.0 + theta*r + 1.0/3.0* theta2 * r^2) * efac
#             #             end
#             #         end
#             #     end
#             # end    
#         else
#             exit()
#         end
#     elseif ktype == "NSMatern" ## Matern Kernel with nu = 5/2
#         tau,sigma,gamma=Theta[1],Theta[2],Theta[3]
#         theta = sqrt(5)/sigma
#         if pder==false 
#             for j =1:length(x)
#                 if x==y 
#                     k1 = j
#                 else 
#                     k1 = 1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     # f_r = exp( gamma * ((1-tx)^2 + (1-ty)^2) )
#                     # f_r = exp( -gamma * (tx+ty)^2 )
#                     f_r = exp( gamma * (tx^2 + 2*tx*ty+ ty^2) )
#                     if kind==0
#                         f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     elseif kind==1    
#                         f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                     elseif kind==2
#                         f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     end
#                     ft=f * f_r
#                     K[j,k]= ft
#                     if x==y && j < k 
#                         K[k,j]= ft
#                     end
#                 end
#             end    
#             return K
#         else 
#             exit()
#         end
#     elseif ktype=="RBF" ##RBF Kernel ## parameter derivative is not implemeted
#         tau,sigma=Theta[1],Theta[2]
#         theta=1.0/sigma
#         if pder==false 
#             K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 if x==y 
#                     k1 = j
#                 else 
#                     k1 = 1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=tau *  exp(-0.5 * (tx-ty)^2 * theta^2 )
#                     elseif kind==1    
#                         f=tau * (tx-ty) * theta^2 *  exp(-0.5* (tx-ty)^2 * theta^2 )
#                     elseif kind==2
#                         f=tau * theta^2 * ( 1 - theta^2 * (ty-tx)^2  )  *  exp(-0.5 * (tx-ty)^2 * theta^2 )
#                     elseif kind==3 
#                         f= tau * ( theta^4 *  (tx-ty)^2 - theta^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
#                     elseif kind==4
#                         f= tau * (tx-ty) * (3 * theta^4 - theta^6 *(tx-ty)^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
#                     elseif kind==5
#                         f= tau * theta^4 * (3- 6 * theta^2 * (tx-ty)^2 + theta^4 * (tx-ty)^4) *  exp(-0.5 * theta^2 * (tx-ty)^2 )
#                     end
#                     K[j,k]=f
#                     if x==y && j < k 
#                         K[k,j]=f
#                     end
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented for RBF Kernel!!");exit()
#         end
#     elseif ktype=="logRBF" ##RBF Kernel ## parameter derivative is not implemeted
#         tau,sigma=Theta[1],Theta[2]
#         theta=1.0/(sigma*sigma)
#         ep = 1.e-30
#         ep = 0.0
#         if pder==false 
#             for j =1:length(x)
#                 if x==y 
#                     k1 = j
#                 else 
#                     k1 = 1
#                 end
#                 if diag == true 
#                     k2 = j
#                 else
#                     k2 = length(y)
#                 end
#                 for k =k1:k2
#                     tx=x[j];ty=y[k]
#                     lx=log(tx+ep);ly=log(ty+ep)                    
#                     r = lx-ly
#                     exfac=exp(-0.5 * abs(r)^2 * theta )
#                     if x[j]==y[k]
#                         r = 0.0
#                         exfac= 1.0
#                     end
#                     if kind==0
#                         f= tau * exfac
#                     elseif kind==1    
#                         f= tau * theta / ty * r * exfac
#                     elseif kind==2
#                         f= tau * theta *( 1-theta* r^2) /(tx*ty) * exfac
#                     elseif kind==3 
#                         f= tau * theta / (ty*ty) * (theta * r^2 - r -1) * exfac
#                     elseif kind==4
#                         f= tau * theta / (tx*ty*ty) *( -theta^2 * r^3 + theta * r^2 + 3*theta*r -1) * exfac
#                     elseif kind==5
#                         f= tau * theta / (tx*tx*ty*ty) * (theta^3 * r^4 - theta*(1+6*theta)*r^2 + 1 + 3*theta) *exfac
#                     end
#                     K[j,k]=f
#                     if x==y && j < k 
#                         K[k,j]=f
#                     end
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented for RBF Kernel!!");exit()
#         end
#     elseif ktype=="Lin*RBF" 
#         tau,sigma,alpha=Theta[1],Theta[2],Theta[3]
#         theta=1.0/sigma
#         ell = theta^2
#         if pder==false 
#             K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 for k =1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f= tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) *( 1 + alpha * tx * ty)
#                     elseif kind==1    
#                         kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 )
#                         f = (alpha * tx - ell * (1+alpha*tx*ty)*(ty-tx)) * kR
#                     elseif kind==2
#                         kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
#                         dkR_dx = - ell * (tx-ty) * kR
#                         dkR_dy = - ell * (ty-tx) * kR
#                         d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         f = alpha * kR + alpha * tx * dkR_dx + alpha * ty * dkR_dy +(1+alpha*tx*ty)*d2kR_dxdy
#                     elseif kind==3 
#                         kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
#                         dkR_dy = - ell * (ty-tx) * kR
#                         d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         f = 2 * alpha * tx * dkR_dy + (1+alpha*tx*ty) * d2kR_dydy
#                     elseif kind==4
#                         kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
#                         dkR_dx = - ell * (tx-ty) * kR
#                         dkR_dy = - ell * (ty-tx) * kR
#                         d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         d2kR_dxdx = d2kR_dydy
#                         d3kR_dxdydy = 2* ell^2 * (tx-ty) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dy
#                         f = 2 * alpha * dkR_dy + alpha * ty * d2kR_dydy + 2 * alpha * tx * dkR_dxdy + (1+alpha*tx*ty)*d3kR_dxdydy
#                     elseif kind==5
#                         kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
#                         dkR_dx = - ell * (tx-ty) * kR
#                         dkR_dy = - ell * (ty-tx) * kR
#                         d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
#                         d2kR_dxdx = d2kR_dydy
#                         d3kR_dxdxdy = 2* ell^2 * (ty-tx) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dy
#                         d3kR_dxdydy = 2* ell^2 * (tx-ty) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dx
#                         d4kR_dxdxdydy = 2* ell^2 * kR + 4* ell^2 * (ty-tx) * dkR_dy + (ell^2 * (tx-ty)^2 -ell) * d2kR_dydy
#                         f = 4*alpha * d2kR_dxdy + 2*alpha*tx*d3kR_dxdxdy + 2*alpha * ty *d3kR_dxdydyx + (1+alpha*tx*ty) *d4kR_dxdxdydy
#                     end
#                     K[j,k]=f
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented for RBF+Lin Kernel!!");exit()
#         end
#     elseif ktype=="Lin+RBF" ##RBF Kernel ## parameter derivative is not implemeted
#         tau,sigma,alpha=Theta[1],Theta[2],Theta[3]
#         theta=1.0/sigma
#         if pder==false 
#             K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 if x==y 
#                     k1=j
#                 else
#                     k1=1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=tau *  exp(-0.5 * (tx-ty)^2 * theta^2 ) + alpha * tx * ty
#                     elseif kind==1    
#                         f=tau * (tx-ty) * theta^2 *  exp(-0.5* (tx-ty)^2 * theta^2 ) + alpha * tx
#                     elseif kind==2
#                         f=tau * theta^2 * ( 1 - theta^2 * (ty-tx)^2  )  *  exp(-0.5 * (tx-ty)^2 * theta^2 ) + alpha
#                     elseif kind==3 
#                         f= tau * ( theta^4 *  (tx-ty)^2 - theta^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
#                     elseif kind==4
#                         f= tau * (tx-ty) * (3 * theta^4 - theta^6 *(tx-ty)^2) * exp(-0.5* (tx-ty)^2 * theta^2 )
#                     elseif kind==5
#                         f= tau * theta^4 * (3- 6 * theta^2 * (tx-ty)^2 + theta^4 * (tx-ty)^4) *  exp(-0.5 * theta^2 * (tx-ty)^2 )
#                     end
#                     K[j,k]=f
#                     if x==y && j < k 
#                         K[k,j]=f
#                     end                        
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented for RBF+Lin Kernel!!");exit()
#         end
#     elseif ktype == "RQ"
#         alpha,sigma=Theta[1],Theta[2]
#         theta = 1.0/(sigma^2)
#         tau = 1.0
#         if pder==false 
#             K =  zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 for k =1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f= tau * ( 1 + theta* (tx-ty)^2 /(2*alpha))^(-alpha)
#                     elseif kind==1    
#                         f= tau * (-(ty-tx)*theta) * ( 1 + theta* (tx-ty)^2 /(2*alpha))^(-alpha-1)
#                     elseif kind==2
#                         tt = ( 1 + theta* (tx-ty)^2 /(2*alpha))
#                         f= tau * tt^(-alpha-2) * ( tt - (alpha+1)*theta/alpha * (tx-ty)^2 )
#                     end
#                     K[j,k]=f
#                 end
#             end    
#             return K
#         else
#             println("pder is not implemented for RQ Kernel");exit()
#         end
#     elseif ktype == "logRQ"
#         alpha,sigma=Theta[1],Theta[2]
#         theta = 1.0/(sigma)
#         tau = 1.0
#         ep = 1.e-30
#         if pder==false 
#             K =  zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 if x==y 
#                     k1 = j
#                 else 
#                     k1 = 1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     lx=log(tx+ep);ly=log(ty+ep)
#                     if x[j]==y[k]
#                         r = 0.0
#                     else
#                         r = lx-ly
#                     end
#                     if kind==0
#                         f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
#                     elseif kind==1    
#                         f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
#                     elseif kind==2
#                         f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
#                     end
#                     K[j,k]=f
#                     if x==y && j < k 
#                         K[k,j]=f
#                     end                        
#                 end
#             end    
#             return K
#         else
#             println("pder is not implemented for RQ Kernel");exit()
#         end
#     elseif ktype=="Lin*Mat"
#         tau,sigma,beta=Theta
#         theta = sqrt(5)/sigma
#         if pder==false 
#             K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 if x==y && length(x)==length(y)
#                     k1=j
#                 else
#                     k1=1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f= (1.0 + beta*tx*ty) * tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     elseif kind==1    
#                         kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                         f=beta * tx * kM + (1.0+beta*tx*ty) *dkdy 
#                     elseif kind==2
#                         kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                         d2kdxdy=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         f = beta * kM + (1.0+beta*tx*ty)*d2kdxdy
#                     end
#                     K[j,k]=f
#                     if j < k && x==y && length(x)==length(y)
#                         K[k,j]=f
#                     end
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented for Lin Kernel!!");exit()
#         end
#     elseif ktype=="Lin+Mat"
#         tau,sigma,beta=Theta
#         theta = sqrt(5)/sigma
#         if pder==false 
#             K = zeros(Float64, length(x), length(y))
#             for j =1:length(x)
#                 if x==y && length(x)==length(y)
#                     k1=j
#                 else
#                     k1=1
#                 end
#                 for k =k1:length(y)
#                     tx=x[j];ty=y[k]
#                     if kind==0
#                         f=  beta*tx*ty + tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                     elseif kind==1    
#                         kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                         f=beta * tx + dkdy 
#                     elseif kind==2
#                         kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
#                         d2kdxdy=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
#                         f = beta + d2kdxdy
#                     end
#                     K[j,k]=f
#                     if j < k && x==y && length(x)==length(y)
#                         K[k,j]=f
#                     end
#                 end
#             end    
#             return K
#         else
#             println("Deribative w.r.t. hyperparameter is not implemented");exit()
#         end
#     end
#     return K
# end
