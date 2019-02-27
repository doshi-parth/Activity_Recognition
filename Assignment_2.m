%% Assignment 2 %%

tic
a1="MyoData";
a2="groundTruth";
c1="fork";
c2="spoon";
e1=".txt";
filewords = [9 10 11 12 13 14 16 17 18 19 21 22 23 24 25 26 27 28 29 30 31 32 33 34 36 37 38 39 40 41];
[~,e]=size(filewords);
accur=[];
col=[];
for f=1:e
    acc=[];
    spoonfileground = "groundTruth/user"+filewords(f)+"/spoon/0.txt";
    spoonfileEMG = "MyoData/user"+filewords(f)+"/spoon/2.txt";
    spoonfileIMU = "MyoData/user"+filewords(f)+"/spoon/1.txt";
    forkfileground = "groundTruth/user"+filewords(f)+"/fork/0.txt";
    forkfileEMG = "MyoData/user"+filewords(f)+"/fork/2.txt";
    forkfileIMU = "MyoData/user"+filewords(f)+"/fork/1.txt";
    col=[col;"user"+filewords(f)];
    M1 = dlmread(fullfile(forkfileground),',');
    N1 = dlmread(fullfile(forkfileIMU),',');
    O1 = dlmread(fullfile(forkfileEMG),',');
    M2 = dlmread(fullfile(spoonfileground),',');
    N2 = dlmread(fullfile(spoonfileIMU),',');
    O2 = dlmread(fullfile(spoonfileEMG),',');
    dataFork1=dataforPCA1(M1,N1,O1);
    dataSpoon1=dataforPCA1(M2,N2,O2);
    dataFork0=dataforPCA0(M1,N1,O1);
    dataSpoon0=dataforPCA0(M2,N2,O2);
    data1=[dataFork1;dataSpoon1];
    data0=[dataFork0;dataSpoon0];
    A=pca(data1);
    K1=data1*A;
    [m1,n1]=size(K1);
    for i=1:m1
        K1(i,n1+1)=1;
    end
    A=pca(data0);
    K2=data0*A;
    [m2,n2]=size(K2);
    for i=1:m2
        K2(i,n2+1)=0;
    end
    split=round(m1*.6);
    training=[K1(1:split,1:5);K2(1:split,1:5)];
    test=[K1(split:m1,1:5);K2(split:m2,1:5)];
    trainLabels=[K1(1:split,n1+1);K2(1:split,n2+1)];
    testLabels=[K1(split:m1,n1+1);K2(split:m2,n2+1)];
    tree=fitctree(training,trainLabels);
    label=predict(tree,test);
    acc=[acc,metrics(label,testLabels)];
    Mdl = fitcsvm(training,trainLabels);
    label = predict(Mdl,test); %assignment 2
    acc=[acc,metrics(label,testLabels)];
    trainnet = transpose(training);
    trainLabels = [trainLabels, int64(~trainLabels)];
    testLabels = [testLabels, int64(~testLabels)];
    tt=double(transpose(trainLabels));
    ts=double(transpose(testLabels));
    net = patternnet(10);
    net.trainParam.showWindow=0;
    net = train(net,trainnet,tt);
    testnet = transpose(test);
    tsn=net(testnet);
    [~,m]=size(tsn);
    label=[];
    for i=1:m
        if tsn(1,i)>tsn(2,i)
            label=[label;1];
        else
            
            label=[label;0];
        end
    end
    acc=[acc,metrics(label,testLabels)];
    accur=[accur;acc];
end
s=["Decision Tree" "SVM" "Neural Network"];
m=["Accuracy" "Precision" "Recall" "F1 Score"];
[~,a]=size(s);
[~,b]=size(m);
h=["User#"];
for i=1:a
    for j=1:b
        h=[h,s(i)+" "+m(j)];
    end
end
header=cellstr(h);
accur=[col,accur];
output = [header; num2cell(accur)];

toc

function r = dataforPCA1(M,N,O)
starttime = round(M(:,1)*50/30,0);
endtime = round(M(:,2)*50/30,0);
[m, n]=size(N);
[~, c]=size(O);
[a, ~] = size(starttime);
d1=[];
for j=1:a
    d=[];
    for i=1:m
        if i>=starttime(j) && i<=endtime(j)
            d=[d;N(i,2:n),O(i,2:c)];
        end
    end
    mm=corrcoef(d);
    [lol1,~]=size(mm);
    if lol1==1
        break;
    end
    cor=[];
    for k=2:18
        for lol=1:k-1
            cor=[cor,mm(k,lol)];
        end
    end
    d1=[d1;reshape(cov(d),1,324),cor,mean(d),std(d),rms(d)];
end
r=d1;
end
function r = dataforPCA0(M,N,O)
starttime = round(M(:,1)*50/30,0);
endtime = round(M(:,2)*50/30,0);
[m, n]=size(N);
[~, c]=size(O);
[a, ~] = size(starttime);
d1=[];
init=1;
for j=1:a
    d=[];
    for i=init:m
        if i>=starttime(j) && i<=endtime(j)
            init=endtime(j);
            break
        end
        d=[d;N(i,2:n),O(i,2:c)];
    end
    mm=corrcoef(d);
    [lol1,~]=size(mm);
    if lol1==1
        break;
    end
    cor=[];
    for k=2:18
        for lol=1:k-1
            cor=[cor,mm(k,lol)];
        end
    end
    d1=[d1;reshape(cov(d),1,324),cor,mean(d),std(d),rms(d)];
end
r=d1;
end

function met=metrics(predLab,actLab)
count=0;
[t1,~]=size(actLab);
for i=1:t1
    if actLab(i)==predLab(i)
        count=count+1;
    end
end
tn=0;
tp=0;
for i=1:t1
    if actLab(i)==0 && predLab(i)==0
        tn=tn+1; %tn
    end
    if actLab(i)==1 && predLab(i)==1
        tp=tp+1; %tp
    end
end
neg=0; %n
pos=0; %p
for i=1:t1
    if actLab(i)==0
        neg=neg+1;
    end
    if actLab(i)==1
        pos=pos+1;
    end
end
falseP=neg-tn;
falseN=pos-tp;
N=neg+pos;
accuracy=(tp+tn)/N;
precision=tp/(tp+falseP);
recall=tp/pos;
f_1=2*((precision*recall)/(precision+recall));
met=[accuracy,precision,recall,f_1];
end