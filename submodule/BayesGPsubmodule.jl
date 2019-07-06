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
    muy=minimum(ytrain) ### for N3LO (more efficient?)
    mstd=std(ytrain)

    xprd=Float64[];pNmax=[]
    unx=collect(xtrain[lt]+2:2.0:xpMax)
    for tmp in unx 
        if (tmp in xtrain)==false 
            push!(xprd,tmp)
        end
    end
    lp=length(xprd)

    return Tsigma,tTheta,xtrain,ytrain,xprd,xun,yun,oxtrain,oytrain,iThetas,lt,lp,Mysigma,muy,mstd
end

function main(xtrain,ytrain,yun,mstep,numN,Mysigma,muy,mstd)
    global achitT,achitY,Tsigma,Pv
    global E0,Evar
    cqT=qT;cqY=qY
    lt=length(xtrain)
    xd=[0.0]; xd2=[0.0]
    ep=5.e-4
    ld=length(xd);ld2=length(xd2)
    
    yd=zeros(Float64,ld)
    yd2=zeros(Float64,ld2)
    
    ep=1.e-30
    
    lp=length(xprd)
    yprd=0*xprd
    
    mujoint=zeros(Float64,lp+ld+ld2);Sjoint=ones(Float64,lp+ld+ld2,lp+ld+ld2)
    Pv=[ [iThetas[i],yprd,yd,yd2,-1.e+300,-1.e+5,-1.e+5,-1.e+5,-1.e+5,-1.e+5,mujoint,Sjoint] for i=1:numN]
    
    ytrain= (ytrain .-muy)/mstd
    NDist=Normal(R,sigR)
    ydel = (ytrain[lt-1]-ytrain[lt])   
    println("xp",xprd," xt ", xtrain," yt ",ytrain," muy ",muy,"mstd",mstd, "lp,ld,ld2", lp," ",ld," ",ld2)
    for tstep=1:mstep
        if tstep<=2;achitT=0;achitY=0;end
        println("tstep ",tstep)
        for ith=1:numN 
            tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
            if tstep == 1
                tyd=0*xd
                tyd2=0*xd2
                tmp=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,yprd,xd,yd,xd2,yd2,iThetas[ith],Monotonic,Convex,"Sample",false)
                tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=tmp
                
                #c_tmp=rand(MvNormal(tmujoint,tSjoint)) ; c_yprd=c_tmp[1:lp]
                
                rtmp = rand(NDist)
                c_tmp= [ ytrain[lt] - ydel * rtmp * (rtmp^(i)-1)/(rtmp-1) for i=1:length(tmujoint)];  c_yprd=c_tmp[1:lp]
                
                tPH = -1.e+15
                if Monotonic==true;c_yd=c_tmp[lp+1:lp+ld]; tyd=c_yd;end
                if Convex==true;c_yd2=c_tmp[lp+ld+1:lp+ld+ld2]; tyd2=c_yd2;end
                Pv[ith]=[iThetas[ith],c_yprd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint] 
                
            else 
                ### Update Theta
                c_Theta = exp.(rand(MvNormal(log.(tTheta),cqT*Tsigma)))
                #while ((c_Theta[1]>1.e+5 || c_Theta[2] > 100.0) && Kernel=="Matern" )|| ( (c_Theta[1]>1.e+5 || c_Theta[2] > 1.e+6) && (Kernel=="logMatern"||Kernel=="logRBF"))
                #while c_Theta[1]>1.e+6 || c_Theta[2] > 100 #&& Kernel=="Matern"
                while c_Theta[1]>1.e+6 || c_Theta[1]<1.e-7 || c_Theta[2] > 100 #&& Kernel=="Matern"
                    c_Theta = exp.(rand(MvNormal(log.(tTheta),cqT*Tsigma)))
                end
                n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2,n_mujoint,n_Sjoint=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,typrd,xd,tyd,xd2,tyd2,c_Theta,Monotonic,Convex,"Sample",false)
                Accept=false
                if n_PH >= tPH 
                    Accept=true
                else
                    lograte=min(0,n_PH-tPH)
                    rtmp = log(rand())
                    if rtmp <=lograte;Accept=true;end
                end
                if Accept==true 
                    tTheta = c_Theta
                    tPH,tlogprior,tllh,tlogpost,tder,tder2=n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2
                    tmujoint,tSjoint=n_mujoint,n_Sjoint
                    achitT += 1
                    #if ith==1;println("Accept T @ith=1");end
                end
                ### Update yprd,yd
                if Monotonic==true
                    yjoint = vcat(typrd,tyd)
                    if Convex==true 
                        yjoint = vcat(yjoint,tyd2)
                    end
                elseif Auxiliary==true
                    yjoint = vcat(typrd,tyd)
                else
                    yjoint = typrd
                end
                if isposdef(cqY*tSjoint)==false 
                    println("eigvals(cqY*tSjoint)",eigvals(cqY*tSjoint))
                    println("isposdef(tSjoint)", isposdef(tSjoint))
                    println("issymmetric(cqY*tSjoint)",issymmetric(cqY*tSjoint))
                end                    
                c_joint = rand(MvNormal(yjoint,cqY*tSjoint)) ## correlated proposal 
                #c_joint = rand(MvNormal(yjoint,Diagonal( (1.e-4) * ones(Float64,length(yjoint),length(yjoint)) ))) ## independent proposal 
                if Monotonic ==true 
                    c_yprd,c_yd=c_joint[1:length(xprd)],c_joint[length(xprd)+1:length(xprd)+length(xd)]
                    if Convex==true 
                        c_yd2=c_joint[lp+ld+1:lp+ld+ld2]
                    else
                        c_yd2=0*xd2
                    end
                elseif Auxiliary==true
                    c_yprd,c_yd=c_joint[1:length(xprd)],c_joint[length(xprd)+1:length(xprd)+length(xd)]
                    c_yd2=xd2*0
                else
                    c_yprd=c_joint;c_yd=0*xd;c_yd2=xd2*0
                end
                Accept=false
                n_PH,n_logprior,n_llh,n_logpost,n_der,nder2,n_mujoint,n_Sjoint=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,c_yprd,xd,c_yd,xd2,c_yd2,tTheta,Monotonic,Convex,"Sample",false)
                Accept=false
                if n_PH > tPH 
                    Accept=true
                else
                    lograte=min(0,n_PH-tPH)
                    rtmp = log(rand())
                    if rtmp <=lograte;Accept=true;end
                end
                if Accept==true 
                    tPH,tlogprior,tllh,tlogpost,tder,tder2=n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2
                    typrd=c_yprd;tyd=c_yd
                    if Convex==true 
                        tyd2=c_yd2
                    else 
                        tyd2=c_yd2
                    end
                    achitY += 1
                    #if ith==1;println("Accept Y @ith=1");end
                end          
                Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
                #                    println("After ith=",ith, " tstep ", tstep ," Pv[ith][2] ", Pv[ith][2])
                if ith == 1 
                    println("c_yprd",c_yprd*mstd.+muy,"\ntmujoint",tmujoint*mstd.+muy, "Accept",Accept)
                    if Monotonic==true || Auxiliary==true
                        # println("delta yd",[ (c_yd[i]-c_yd[i-1]) for i=2:ld])
                        # println("delta yd/xd",[ (c_yd[i]-c_yd[i-1])/xd[i-1] for i=2:ld])
                        # println("delta2 yd",[ (c_yd[i]-2*c_yd[i-1]+c_yd[i-2]) for i=3:ld])
                        if Convex==true 
                            println("c_yd2",c_yd2)
                        end
                    end
                end
            end  
        end
        ### ~Resampling~
        if  tstep == 200
            Pv=Resample(Pv,numN)
            cqY = qY * qYfac
        end
        ####### Adaptive proposals
        # if  500 > tstep > 200
        #     lqT=log(cqT);lqY=log(cqY)
        #     b_n = bn(0.1*tstep)
        #     lqT = lqT + b_n * (achitT/(numN*tstep) - rMH[1]) 
        #     lqY = lqY + b_n * (achitY/(numN*tstep) - rMH[2]) 
        #     cqT = exp(lqT)
        #     cqY = exp(lqY)
        # end
        if tstep>1
            s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Accept. ratio  T:" 100*achitT/(numN*(tstep-1)) " Y:" 100*achitY/(numN*(tstep-1)) " qT,qY= " cqT cqY
            println(s,"\n")
        end    
        Summary(tstep,numN,xtrain,ytrain,xun,yun,xprd,xd,Pv,muy,mstd,"R")
        E0,Evar=Summary(tstep,numN,xtrain,ytrain,xun,yun,xprd,xd,Pv,muy,mstd,"Theta")
    end
    iot = open("Thetas_"*string(inttype)*".dat", "w")
    for ith = 1:numN
        println(iot,Pv[ith][1][1]," ",Pv[ith][1][2]," ",Pv[ith][5])
    end
    close(iot)
    return [E0,Evar]
end

function detR(xtrain,ytrain,yun,mstep,Mysigma,muy,mstd)
    global achitT,achitY,Tsigma,Pv
    global Wm, Ws
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
    
    mujoint=zeros(Float64,lp+ld+ld2);Sjoint=ones(Float64,lp+ld+ld2,lp+ld+ld2)
    iThetas=[ [100*rand(),30.0*rand()] for i=1:tnumN]
    Pv=[ [iThetas[i],yprd,yd,yd2,-1.e+300,-1.e+5,-1.e+5,-1.e+5,-1.e+5,-1.e+5,mujoint,Sjoint] for i=1:tnumN]
    
    ytrain= (ytrain .-muy)/mstd
    for tstep=1:Rstep
        if tstep<=2;achitT=0;achitY=0;end
        for ith=1:tnumN 
            tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[ith]
            if tstep == 1                
                tyd=0*xd
                tyd2=0*xd2
                tmp=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,yprd,xd,yd,xd2,yd2,iThetas[ith],Monotonic,Convex,"Sample",true)
                tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=tmp
                #c_tmp=rand(MvNormal(tmujoint,1.0*tSjoint))
                c_tmp=tmujoint
                c_yprd=c_tmp[1:lp]
                
                tPH= -1.e+20
                if Monotonic==true;c_yd=c_tmp[lp+1:lp+ld]; tyd=c_yd;end
                if Convex==true;c_yd2=c_tmp[lp+ld+1:lp+ld+ld2]; tyd2=c_yd2;end
                Pv[ith]=[iThetas[ith],c_yprd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint] 
            else 
                ### Update Theta
                c_Theta = exp.(rand(MvNormal(log.(tTheta),cqT*Tsigma)))
                #while ((c_Theta[1]>1.e+5 || c_Theta[2] > 2) && Kernel=="Matern") || ( (c_Theta[1]>1.e+5 || c_Theta[2] > 1.e+3) && (Kernel=="logMatern" ||Kernel=="logRBF" ))
                while c_Theta[1]>1.e+6 || c_Theta[2] > 100#c_Theta[2] > 200)# && Kernel=="Matern"
                    c_Theta = exp.(rand(MvNormal(log.(tTheta),cqT*Tsigma)))
                end
                n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2,n_mujoint,n_Sjoint=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,typrd,xd,tyd,xd2,tyd2,c_Theta,Monotonic,Convex,"Sample",true)
                Accept=false
                if n_PH >= tPH 
                    Accept=true
                else
                    lograte=min(0,n_PH-tPH)
                    rtmp = log(rand())
                    if rtmp <=lograte;Accept=true;end
                end
                if Accept==true 
                    tTheta = c_Theta
                    tPH,tlogprior,tllh,tlogpost,tder,tder2=n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2
                    tmujoint,tSjoint=n_mujoint,n_Sjoint
                    achitT += 1
                end
                ### Update yprd,yd
                if Monotonic==true
                    yjoint = vcat(typrd,tyd)
                    if Convex==true 
                        yjoint = vcat(yjoint,tyd2)
                    end
                elseif Auxiliary==true
                    yjoint = vcat(typrd,tyd)
                else
                    yjoint = typrd
                end
                c_joint = rand(MvNormal(yjoint,cqY*tSjoint))
                #if ith==1;println("yjoint",yjoint,"\nc_joint",c_joint);end
                if Monotonic ==true 
                    c_yprd,c_yd=c_joint[1:length(xprd)],c_joint[length(xprd)+1:length(xprd)+length(xd)]
                    if Convex==true 
                        c_yd2=c_joint[lp+ld+1:lp+ld+ld2]
                    else
                        c_yd2=0*xd2
                    end
                elseif Auxiliary==true
                    c_yprd,c_yd=c_joint[1:length(xprd)],c_joint[length(xprd)+1:length(xprd)+length(xd)]
                    c_yd2=xd2*0
                else
                    c_yprd=c_joint;c_yd=0*xd;c_yd2=xd2*0
                end
                Accept=false
                n_PH,n_logprior,n_llh,n_logpost,n_der,nder2,n_mujoint,n_Sjoint=oneGP(Kernel,tstep,mstep,ith,xtrain,ytrain,Mysigma,muy,mstd,xprd,c_yprd,xd,c_yd,xd2,c_yd2,tTheta,Monotonic,Convex,"Sample",true)
                Accept=false
                if n_PH > tPH 
                    Accept=true
                else
                    lograte=min(0,n_PH-tPH)
                    rtmp = log(rand())
                    if rtmp <=lograte;Accept=true;end
                end
                if Accept==true 
                    tPH,tlogprior,tllh,tlogpost,tder,tder2=n_PH,n_logprior,n_llh,n_logpost,n_der,n_der2
                    typrd=c_yprd;tyd=c_yd
                    if Convex==true 
                        tyd2=c_yd2
                    else 
                        tyd2=c_yd2
                    end
                    achitY += 1
                end          
                Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
                
                if ith == 1 
                    #if Accept == true;println("c_yprd",c_yprd*mstd.+muy," tmujoint[:lp] ",tmujoint[1:lp], " Accept",Accept);end
                    if Monotonic==true || Auxiliary==true
                        if Convex==true 
                            println("c_yd2",c_yd2)
                        end
                    end
                end
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
            s = @sprintf "%s %9.2f %s %9.2f %s %9.3e %9.3e " "Acchit T:" 100*achitT/(tnumN*(tstep-1)) " Y:" 100*achitY/(tnumN*(tstep-1)) " qT,qY= " cqT cqY
            println(s)
            Wm,Ws=Summary(tstep,tnumN,xtrain,ytrain,xun,yun,xprd,xd,Pv,muy,mstd,"R")       
        end    
    end    
    return Wm,Ws
end
function KernelMat(ktype,Theta,x,y,kind,Mat,pder=false) 
    if ktype == "Matern" ## Matern Kernel with nu = 5/2
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(5)/sigma
        if pder==false 
            K = zeros(Float64, length(x), length(y))
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
            return K
        else
            K = zeros(Float64, length(x), length(y))
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
            tmp,cL,cLinv,Ginv,logdetK = Mchole(G,length(x))                
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
            K = zeros(Float64, length(x), length(y))
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
        ep = 1.e-30
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
                    lx=log(tx+ep);ly=log(ty+ep)
                    if x[j]==y[k]
                        r = 0.0
                        efac=1.0
                    else
                        r = lx-ly
                        efac=exp(-theta*abs(r))
                    end
                    if kind==0
                        f=tau * ( 1 + theta*abs(r) ) * efac 
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
            return K
        else
            exit()
        end
    elseif ktype == "logMatern"
        tau,sigma=Theta[1],Theta[2]
        theta = sqrt(5)/sigma
        ep = 1.e-30
        ep = 0.0
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
                    lx=log(tx+ep);ly=log(ty+ep)
                    if x[j]==y[k]
                        r = 0.0
                        efac=1.0
                    else
                        r = lx-ly
                        efac=exp(-theta*abs(r))
                    end
                    if kind==0
                        f=tau * ( 1 + theta*abs(r) + 1.0/3.0* theta * theta * r^2) * efac
                    elseif kind==1    
                        #f=-tau * 1.0/3.0*theta*theta * abs(r) * (1+theta*abs(r)) * efac * sign(-r)
                        f = -tau* theta^2 / 3.0 * sign(ty-tx) * efac * (abs(r)* theta * abs(r)^2)  / (ty+ep)
                    elseif kind==2
                        f=tau/3.0 * theta * theta * (1 +theta*abs(r)- theta^2 * r^2) * efac / ((tx+ep)*(ty+ep))
                    end
                    K[j,k]=f
                    if x==y && j < k 
                        K[k,j]=f
                    end
                end
            end    
            return K
        else
            exit()
        end
    elseif ktype == "NSMatern" ## Matern Kernel with nu = 5/2
        tau,sigma,gamma=Theta[1],Theta[2],Theta[3]
        theta = sqrt(5)/sigma
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
                    #                    f_r = exp( gamma * ((1-tx)^2 + (1-ty)^2) )
                    #                    f_r = exp( -gamma * (tx+ty)^2 )
                    f_r = exp( gamma * (tx^2 + 2*tx*ty+ ty^2) )
                    if kind==0
                        f=tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    elseif kind==1    
                        f=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                    elseif kind==2
                        f=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                    end
                    ft=f * f_r
                    K[j,k]= ft
                    if x==y && j < k 
                        K[k,j]= ft
                    end
                end
            end    
            return K
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
                    lx=log(tx+ep);ly=log(ty+ep)
                    r = lx-ly
                    exfac=exp(-0.5 * r^2 * theta )
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
    elseif ktype=="logRBF1" ##RBF Kernel ## parameter derivative is not implemeted
        tau,sigma=Theta[1],Theta[2]
        theta=1.0/sigma
        ep = 1.e-30
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
                    lx=log(tx+ep);ly=log(ty+ep)
                    r = abs(lx-ly)
                    exfac=exp(-0.5 * r * theta )
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
    elseif ktype=="Lin*RBF" 
        tau,sigma,alpha=Theta[1],Theta[2],Theta[3]
        theta=1.0/sigma
        ell = theta^2
        if pder==false 
            K = zeros(Float64, length(x), length(y))
            for j =1:length(x)
                for k =1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f= tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) *( 1 + alpha * tx * ty)
                    elseif kind==1    
                        kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 )
                        f = (alpha * tx - ell * (1+alpha*tx*ty)*(ty-tx)) * kR
                    elseif kind==2
                        kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
                        dkR_dx = - ell * (tx-ty) * kR
                        dkR_dy = - ell * (ty-tx) * kR
                        d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
                        f = alpha * kR + alpha * tx * dkR_dx + alpha * ty * dkR_dy +(1+alpha*tx*ty)*d2kR_dxdy
                    elseif kind==3 
                        kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
                        dkR_dy = - ell * (ty-tx) * kR
                        d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
                        f = 2 * alpha * tx * dkR_dy + (1+alpha*tx*ty) * d2kR_dydy
                    elseif kind==4
                        kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
                        dkR_dx = - ell * (tx-ty) * kR
                        dkR_dy = - ell * (ty-tx) * kR
                        d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
                        d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
                        d2kR_dxdx = d2kR_dydy
                        d3kR_dxdydy = 2* ell^2 * (tx-ty) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dy
                        f = 2 * alpha * dkR_dy + alpha * ty * d2kR_dydy + 2 * alpha * tx * dkR_dxdy + (1+alpha*tx*ty)*d3kR_dxdydy
                    elseif kind==5
                        kR = tau *  exp(-0.5* (tx-ty)^2 * theta^2 ) 
                        dkR_dx = - ell * (tx-ty) * kR
                        dkR_dy = - ell * (ty-tx) * kR
                        d2kR_dxdy = (ell - ell^2 * (ty-tx)^2 -ell) * kR
                        d2kR_dydy = -(ell - ell^2 * (ty-tx)^2 -ell) * kR
                        d2kR_dxdx = d2kR_dydy
                        d3kR_dxdxdy = 2* ell^2 * (ty-tx) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dy
                        d3kR_dxdydy = 2* ell^2 * (tx-ty) * kR + (ell^2 * (tx-ty)^2 * - ell) * dkR_dx
                        d4kR_dxdxdydy = 2* ell^2 * kR + 4* ell^2 * (ty-tx) * dkR_dy + (ell^2 * (tx-ty)^2 -ell) * d2kR_dydy
                        f = 4*alpha * d2kR_dxdy + 2*alpha*tx*d3kR_dxdxdy + 2*alpha * ty *d3kR_dxdydyx + (1+alpha*tx*ty) *d4kR_dxdxdydy
                    end
                    K[j,k]=f
                end
            end    
            return K
        else
            println("Deribative w.r.t. hyperparameter is not implemented for RBF+Lin Kernel!!");exit()
        end
    elseif ktype=="Lin+RBF" ##RBF Kernel ## parameter derivative is not implemeted
        tau,sigma,alpha=Theta[1],Theta[2],Theta[3]
        theta=1.0/sigma
        if pder==false 
            K = zeros(Float64, length(x), length(y))
            for j =1:length(x)
                if x==y 
                    k1=j
                else
                    k1=1
                end
                for k =k1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f=tau *  exp(-0.5 * (tx-ty)^2 * theta^2 ) + alpha * tx * ty
                    elseif kind==1    
                        f=tau * (tx-ty) * theta^2 *  exp(-0.5* (tx-ty)^2 * theta^2 ) + alpha * tx
                    elseif kind==2
                        f=tau * theta^2 * ( 1 - theta^2 * (ty-tx)^2  )  *  exp(-0.5 * (tx-ty)^2 * theta^2 ) + alpha
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
            println("Deribative w.r.t. hyperparameter is not implemented for RBF+Lin Kernel!!");exit()
        end
    elseif ktype == "RQ"
        alpha,sigma=Theta[1],Theta[2]
        theta = 1.0/(sigma^2)
        tau = 1.0
        if pder==false 
            K =  zeros(Float64, length(x), length(y))
            for j =1:length(x)
                for k =1:length(y)
                    tx=x[j];ty=y[k]
                    if kind==0
                        f= tau * ( 1 + theta* (tx-ty)^2 /(2*alpha))^(-alpha)
                    elseif kind==1    
                        f= tau * (-(ty-tx)*theta) * ( 1 + theta* (tx-ty)^2 /(2*alpha))^(-alpha-1)
                    elseif kind==2
                        tt = ( 1 + theta* (tx-ty)^2 /(2*alpha))
                        f= tau * tt^(-alpha-2) * ( tt - (alpha+1)*theta/alpha * (tx-ty)^2 )
                    end
                    K[j,k]=f
                end
            end    
            return K
        else
            println("pder is not implemented for RQ Kernel");exit()
        end
    elseif ktype == "logRQ"
        alpha,sigma=Theta[1],Theta[2]
        theta = 1.0/(sigma)
        tau = 1.0
        ep = 1.e-30
        if pder==false 
            K =  zeros(Float64, length(x), length(y))
            for j =1:length(x)
                if x==y 
                    k1 = j
                else 
                    k1 = 1
                end
                for k =k1:length(y)
                    tx=x[j];ty=y[k]
                    lx=log(tx+ep);ly=log(ty+ep)
                    if x[j]==y[k]
                        r = 0.0
                    else
                        r = lx-ly
                    end
                    if kind==0
                        f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
                    elseif kind==1    
                        f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
                    elseif kind==2
                        f= ( 1 + theta* r^2 /(2*alpha))^(-alpha)
                    end
                    K[j,k]=f
                    if x==y && j < k 
                        K[k,j]=f
                    end                        
                end
            end    
            return K
        else
            println("pder is not implemented for RQ Kernel");exit()
        end
elseif ktype=="Lin*Mat"
tau,sigma,beta=Theta
theta = sqrt(5)/sigma
if pder==false 
    K = zeros(Float64, length(x), length(y))
    for j =1:length(x)
        if x==y && length(x)==length(y)
            k1=j
        else
            k1=1
        end
        for k =k1:length(y)
            tx=x[j];ty=y[k]
            if kind==0
                f= (1.0 + beta*tx*ty) * tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
            elseif kind==1    
                kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                f=beta * tx * kM + (1.0+beta*tx*ty) *dkdy 
            elseif kind==2
                kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                d2kdxdy=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                f = beta * kM + (1.0+beta*tx*ty)*d2kdxdy
            end
            K[j,k]=f
            if j < k && x==y && length(x)==length(y)
                K[k,j]=f
            end
        end
    end    
    return K
else
    println("Deribative w.r.t. hyperparameter is not implemented for Lin Kernel!!");exit()
end
elseif ktype=="Lin+Mat"
tau,sigma,beta=Theta
theta = sqrt(5)/sigma
if pder==false 
    K = zeros(Float64, length(x), length(y))
    for j =1:length(x)
        if x==y && length(x)==length(y)
            k1=j
        else
            k1=1
        end
        for k =k1:length(y)
            tx=x[j];ty=y[k]
            if kind==0
                f=  beta*tx*ty + tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
            elseif kind==1    
                kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                f=beta * tx + dkdy 
            elseif kind==2
                kM = tau * ( 1 + theta*abs(tx-ty) + 1.0/3.0* theta * theta *(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                dkdy=-tau * 1.0/3.0*theta*theta * abs(tx-ty) * (1+theta*abs(tx-ty)) * exp(-theta*abs(tx-ty)) * sign(ty-tx)
                d2kdxdy=tau/3.0 * theta * theta * (1 +theta*abs(tx-ty)- theta*theta*(tx-ty)^2) * exp(-theta*abs(tx-ty)) 
                f = beta + d2kdxdy
            end
            K[j,k]=f
            if j < k && x==y && length(x)==length(y)
                K[k,j]=f
            end
        end
    end    
    return K
else
    println("Deribative w.r.t. hyperparameter is not implemented");exit()
end
end
end
function Mchole(tmpA,ln) 
    cL=cholesky(tmpA, Val(false); check = true)
    cLinv=inv(cL.L)
    Ainv=transpose(cLinv)*cLinv
    global logLii
    logLii=0.0
    for i in 1:ln
        logLii = logLii + log(cL.L[i,i])
    end
    return tmpA,cL,cLinv,Ainv, 2.0*logLii
end
function nu_t(tstep,maxstep) 
    fac = 1.e+1
    fac = 1.e+2
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
    global hit1,hit2
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

# function singleGP(Theta,xt,yt)
#     xprd = collect(0:0.05:0.2)
#     Ktt=KernelMat(Kernel,Theta,xt,xt,0,true) + Mysigma
#     Kpp=KernelMat(Kernel,Theta,xprd,xprd,0,true) 
#     Kpt=KernelMat(Kernel,Theta,xprd,xt,0,true) 
#     Ktt,cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))
#     A = Kpp; C = Kpt
#     mu_y = reshape([ 0.0 for i=1:length(xprd)],length(xprd))
#     mu_yt= reshape([ 0.0 for i=1:length(xt)],length(xt))
#     mujoint = mu_y + C*Kttinv*(yt-mu_yt)
#     tmp=cLinv*transpose(C)
#     Sjoint = A - transpose(tmp)*tmp
#     Sjoint,Sj_cL,Sj_cLinv,Sjinv,logdetSj = Mchole(Sjoint,length(xprd)) 
#     means=mujoint
#     stds=[ sqrt(Sjoint[i,i]) for i=1:length(xprd)]
#     return [xprd,means,stds]
# end

# function LAsample(ThetaMAP,Kernel)
#     tau,sigma=ThetaMAP
#     tmp=KernelMat(Kernel,ThetaMAP,xtrain,xtrain,0,true,true)
#     K,d2Hdt2,d2Hds2,d2Hdtds=tmp
#     Hessian = -[d2Hdt2 d2Hdtds; d2Hdtds d2Hds2]
#     tmp,cL,cLinv,Hinv,logdet = Mchole(Hessian,length(ThetaMAP))
#     c_Thetas=[ ThetaMAP for i=1:numN]
#     for i=1:numN      
#         c_Thetas[i] = rand(MvNormal(ThetaMAP,Hinv))
#         while any(c_Thetas[i].<0)
#             c_Thetas[i] = rand(MvNormal(ThetaMAP,Hinv))
#         end
#     end
#     return c_Thetas
# end
function Summary(tstep,numN,xt,yt,xun,yun,xprd,xd,Pv,muy,mstd,plttype)   
    global bestPH,bestV,bestW, Wm, Ws
    lt=length(xt); lp=length(xprd); ld=length(xd)
    yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprds= [ [0.0 for ith=1:numN] for kk=1:lp]
    w_yprd2s= [ [0.0 for ith=1:numN] for kk=1:lp]
    yds= [ [0.0 for ith=1:numN] for kk=1:ld]
    Weights= [0.0 for ith=1:numN]    
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
        
        logw=tlogprior+tllh  +tder +tlogpost + tder2
        
        if logw > 709 
            Weights[ith] = 1.e+30
        elseif logw< -746
            Weights[ith] = 1.0e-100
        else
            Weights[ith] = exp(logw)
        end                
        #        if ith==1; println(Weights[ith]);end
        for kk=1:length(xprd)
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
    w_yprds = w_yprds/sumW
    w_yprd2s = w_yprd2s/sumW       
    
    means=[ 0.0 for i=1:length(xprd)]
    stds=[ 0.0 for i=1:length(xprd)]
    for kk=1:length(xprd)
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
        Nmin=string( Int64(oxtrain[1])) 
        Nmax=string( Int64(oxtrain[length(oxtrain)]))
        fn="Posterior_"*string(inttype)*".dat"
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
            s=@sprintf "%20.8e" log.(Weights[ith]/sumW)
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
        deno=oytrain[lt-1]-oytrain[lt]
        nume=oytrain[lt] .- yprds[1]
        Dratio = nume ./deno
        Wm,Ws=weightedmean(Dratio,Weights/sumW)
        println("R(mean)=$Wm, Rstd=$Ws")
        return Wm, Ws
    end       
    return means[lp],stds[lp]
end


function oneGP(Kernel,tstep,mstep,ith,xt::T,yt::T,Mysigma,muy,mstd,xprd::T,yprd::T,xd::T,yd::T,xd2::T,yd2::T,Theta,Monotonic,Convex,GPmode,fixR) where {T<:AbstractArray{Float64,1}}
    lp=length(xprd);ld=length(xd);ld2=length(xd2);lt=length(xt)
    lj=lt
    if Monotonic==true; lj += ld ;end
    if Convex==true;lj += ld2 ;end
    if GPmode=="Sample"
        if Monotonic==true ## Not complete
            Ktt=KernelMat(Kernel,Theta,xt,xt,0,true) + Diagonal( (1.e-8) * ones(Float64,lt,lt))
            Kpp=KernelMat(Kernel,Theta,xprd,xprd,0,true) + Diagonal( (1.e-8) * ones(Float64,lp,lp))
            Kpt=KernelMat(Kernel,Theta,xprd,xt,0,true) 
            Kdd=KernelMat(Kernel,Theta,xd,xd,2,true) + Diagonal( (1.e-8) * ones(Float64,ld,ld))
            Kpd=KernelMat(Kernel,Theta,xprd,xd,1,true) 
            Ktd=KernelMat(Kernel,Theta,xt,xd,1,true) 
            Ktt,cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))
            
            if Convex==true 
                K02=KernelMat(Kernel,Theta,xt,xd2,3,true) 
                Kp2=KernelMat(Kernel,Theta,xprd,xd2,3,true) 
                K12=KernelMat(Kernel,Theta,xd,xd2,4,true) 
                ld2=length(xd2)
                K22=KernelMat(Kernel,Theta,xd2,xd2,5,true) + Diagonal( (1.e-4) * ones(Float64,ld2,ld2))
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
            mu_y = reshape([ 0.0 for i=1:length(xprd)],length(xprd))
            mu_yt= reshape([ 0.0 for i=1:length(xt)],length(xt))
            mu_d = reshape([ 0.0 for i=1:length(xd)],length(xd))
            if Convex==true
                mu_d2= reshape([ 0.0 for i=1:length(xd2)],length(xd2))
                mujoint = vcat(vcat(mu_y,mu_d),mu_d2)
            else
                mujoint = vcat(mu_y,mu_d)
            end
            mujoint += C*Kttinv*(yt-mu_yt)
            
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
                Sjoint = Sjoint + Diagonal( 10*abs(minEV) * ones(Float64,length(mujoint),length(mujoint)))
                if errc > 100
                    println("NotPSD:Sjoint: eigvals",eigvals(Sjoint), " @ ",Theta, "isposdef(Sjoint)", isposdef(Sjoint) )
                end
            end
            if Convex==true
                Sjoint,Sj_cL,Sj_cLinv,Sjinv,logdetSj = Mchole(Sjoint,length(xprd)+length(xd)+length(xd2)) 
            else
                Sjoint,Sj_cL,Sj_cLinv,Sjinv,logdetSj = Mchole(Sjoint,length(xprd)+length(xd))
            end
            
            Sjoint,Sj_cL,Sj_cLinv,Sjinv,logdetSj = Mchole(Sjoint,length(xprd)+length(xd)) 
        else
            Ktt=KernelMat(Kernel,Theta,xt,xt,0,true) + Mysigma
            Kpp=KernelMat(Kernel,Theta,xprd,xprd,0,true)            
            Kpt=KernelMat(Kernel,Theta,xprd,xt,0,true) 
            if isposdef(Ktt) == false 
                println("issymmetric(Ktt)",issymmetric(Ktt), " @Theta=", Theta)
                println("eigvals:",eigvals(Ktt))
            end
            Ktt,cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))
            A = Kpp; C = Kpt
            mu_y = reshape([ 0.0 for i=1:lp],lp)
            mu_yt= reshape([ 0.0 for i=1:lt],lt)
            mujoint = mu_y + C*Kttinv*(yt-mu_yt)
            tmp=cLinv*transpose(C)
            
            Sjoint = A - transpose(tmp)*tmp
            errc = 0
            while isposdef(Sjoint)==false
                ttmp=eigvals(Sjoint)
                minEV=ttmp[1]              
                Sjoint = Sjoint + Diagonal( 100*abs(minEV) * ones(Float64,lp,lp))
                errc += 1
                if errc > 100
                    println("NotPSD:Sjoint L516:@",Theta,ttmp)                    
                end
            end 
            Sjoint,Sj_cL,Sj_cLinv,Sjinv,logdetSj = Mchole(Sjoint,length(xprd)) 
        end
    else
        tmp=KernelMat(Kernel,Theta,xt,xt,0,true)
        Ktt=tmp 
        Ktt,cL,cLinv,Kttinv,logdetK = Mchole(Ktt,length(xt))
        mu_yt= reshape([ 0.0 for i=1:length(xt)],length(xt))
    end
    logprior = 0.0 ## uniform 
    
    term1 = -0.5*transpose(yt-mu_yt)*Kttinv*(yt-mu_yt)
    term2 = -0.5*logdetK
    llh = term1 + term2 
    der = 0.0
    der2 = 0.0
    nu  = nu_t(tstep,mstep)
    
    if Monotonic==true 
        for kk=1:length(yd)
            tmp = Phi(nu*yd[kk])
            if tmp == 0.0 
                der += - 1.e+8
            else 
                der += log(tmp)
            end
        end
        if Convex==true 
            convcost=nu #*1.e+1
            for kk=1:length(yd2)
                tmp = Phi(convcost*yd2[kk])
                if tmp == 0.0
                    der2 += -1.e+8
                else 
                    der2 += log(tmp)
                end
            end
        end
    end
    
    ###auxilial derivative observation    
    
    if Monotonic==false 
        nupen=1.e+10
        tmp = Phi( nu * (yt[lt]-yprd[1])*mstd)
        if ith == 1 && printder==true && fixR==false
            s=@sprintf "%s %8.3f %s %8.3f %s %8.3f %10.4e"  "xprd[lp]" xprd[lp]  "yprd[1]" yprd[1]*mstd+muy "<- yt[lt]" yt[lt]*mstd+muy log(tmp)
            println(s)
        end
        if tmp == 0.0
            der += -nupen
        else 
            der += log(tmp)
        end
        if lp > 1
            for k =1:lp-1
                tmp = Phi( nu*(yprd[k]-yprd[k+1])*mstd)
                if ith ==1 && printder==true && fixR==false
                    s=@sprintf "%s %8.3f %s  %8.3f %s %8.3f %s %10.4e" "xprd=" xprd[k] "yprd[k]" yprd[k]*mstd+muy "->" yprd[k+1]*mstd+muy  "log(tmp)" log(tmp)
                    println(s)
                end
                if tmp == 0.0
                    der += -nupen
                else 
                    der += log(tmp)
                end
            end
        end
    end
    if CImode=="NCSM" && fixR==false
        convcost = nu #* 1.e-1
        convpen = 1.e+8
        
        if xprd[1]> 0 
            tR = (yt[lt]-yprd[1])/(yt[lt-1]-yt[lt])
            U_idealR=R+sigfac*sigR; L_idealR=R-sigfac*sigR
            #### Non Gaussian 
            tmp = Phi( convcost * (U_idealR-tR) ) #+ Phi( convcost * (tR-L_idealR) )     
            if tmp == 0.0
                der2 += -convpen 
            else 
                der2 += log(tmp)
            end            
            ### Gaussian 
            ##der2 += - 0.5 * (tR-R)^2 / (sigR*sigR)                
            
            if ith == 1 && printder==true && fixR==false
                println("For der2")
                s=@sprintf "%s %8.3f %s %8.3f" "y(t-1)-y(t)" mstd*(yt[lt-1]-yt[lt]) "y(t)-yprd[1]" mstd*(yt[lt]-yprd[1]) 
                print(s)
                s=@sprintf "%s %10.3f %s %10.3e %10.3e %s %10.3e" " tR" tR "R(L,U)" L_idealR U_idealR "Cost" tmp
                print(s)
                println("(U_idealR-tR)",U_idealR-tR)
                println("(tR-L_idealR)",tR-L_idealR)
            end
        end
        if lp > 1 
            tR = (yprd[1]-yprd[2])/(yt[lt]-yprd[1]) 
            Nk=Int64(xprd[2])
            Nj=Int64(xprd[1])
            Nt=Int64(xt[lt])
            Ni=Nt
            
            if abs(Nj-Nt) == 2 && abs(Nk-Nj)==2
                U_idealR=R+sigfac*sigR; L_idealR=R-sigfac*sigR 
                tmp = Phi( convcost * (U_idealR-tR) ) #+ Phi( convcost * (tR-L_idealR) )     
            else
                tmp = 1.0
            end
            if ith==1 && printder==true && fixR==false
                s=@sprintf "%s %8.3f %s %8.3f %10.4e %s %10.3e %s %10.3e %10.3e" "yt-yp[1]" (yt[lt]*-yprd[1])*mstd "yp[1]-yp[2]" (yprd[1]-yprd[2])*mstd tmp "tR" tR "idealR(L,U)" L_idealR U_idealR
                println(s)
            end
            if tmp == 0.0
                der2 += -convpen
            else 
                der2 += log(tmp)
            end                 
            ### Gaussian 
            # der2 += - 0.5 * (tR-R)^2 / (sigR*sigR)
            
            if lp>2 
                for k =1:lp-2
                    tR = (yprd[k+1]-yprd[k+2])/(yprd[k]-yprd[k+1])
                    
                    Nk=Int64(xprd[k+2])
                    Nj=Int64(xprd[k+1])
                    Ni=Int64(xprd[k])
                    if abs(Nj-Ni) == 2 && abs(Nk-Nj)==2
                        U_idealR=R+sigfac*sigR; L_idealR=R-sigfac*sigR
                        tmp = Phi( convcost * (U_idealR-tR) ) 
                    else
                        tmp = 1.0
                    end
                    if ith ==1 && printder==true && fixR==false
                        s=@sprintf "%s %8.3f %s %8.3f %s %10.3e %10.3e %10.3e" "Delta y"  (yprd[k+1]-yprd[k+2])*mstd "<-" (yprd[k]-yprd[k+1])*mstd "tR,idealR(Lower,Upper)" tR L_idealR U_idealR
                        print(s)
                        s=@sprintf "%s %10.3e %10.3e" "Cost(L,U)" (tR-L_idealR) (U_idealR-tR)
                        println(s)
                    end
                    if tmp == 0.0
                        der2 += -convpen / (xprd[k+2]^2)
                    else 
                        der2 += log(tmp)
                    end
                end
            end
        end
    end
    
    #### auxilial derivative observation
    
    if GPmode =="Sample"
        if Monotonic==true 
            yjoint = vcat(yprd,yd)
            if Convex==true 
                yjoint = vcat(yjoint,yd2)
            end
        else
            yjoint = yprd
        end
        term1 = -0.5 * transpose(yjoint-mujoint)*Sjinv*(yjoint-mujoint)
        term2 = -0.5 * logdetSj
        if term1 > 0 ## This may occur if sigma is "unphysical" (breaking PSD)
            term1 = term1 - 1.e+15
            println("warn:negative term1")
        end
        logpost = term1 + term2
        PH = logprior + llh + logpost + der + der2
        if ith == 1 && fixR==false
            s = @sprintf "%s %10.3e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" PH "   llh:" llh "   logpost:" logpost "   der:" der "der2:" der2
            println(s,"@",Theta)
            # elseif Theta[1]>30 && ith <50 && tstep>20
            #     s = @sprintf "%s %10.2e %s %10.2e %s %10.2e %s %10.2e %s %10.2e" "PH:" PH "   llh:" llh "   logpost:" logpost "   der:" der "der2:" der2
            #     #println("unphys?:",s," @",Theta)
        end
    else
        PH = logprior + llh
    end
    
    if GPmode=="Sample"
        return PH,logprior,llh,logpost,der,der2,mujoint,Sjoint
    else
        return PH,logprior,llh
    end
end

function Resample(Pv,numN)
    ###Pv[ith]=[tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint]
    w_der=[0.0 for i=1:numN]
    x=[ [] for i=1:numN]
    for i =1:numN
        tTheta,typrd,tyd,tyd2,tPH,tlogprior,tllh,tlogpost,tder,tder2,tmujoint,tSjoint=Pv[i]
        tmp= tder + tder2
        if tmp > 709 
            tmp= 1.e+30
        elseif tmp<-746
            tmp=1.0e-100
        else
            tmp=exp(tmp)
        end
        w_der[i]=tmp
    end
    StatsBase.alias_sample!(Pv,weights(w_der),x)
    return x
end                    
