%%
close all;
f1=11; f2=5; %frequencies of two signals;
tt=0:10:1000;
f_sum=sin(2*pi*tt)+sin(1/(2*pi)*tt);%f_sum=sawtooth(1/(2*pi)*tt)+square(1/(2*pi)*sqrt(11)*tt);%
plot(tt,f_sum,'b')
%commulative frequency is 11/11=1; (a*x1=b*x2; f_c=x1/b=x2/a);
 g = fittype('a+b*x+c*x*x');
 model=fit(tt',f_sum',g);
 plot(tt,f_sum,':',tt,model.a+model.b*tt+model.c*tt.*tt,'g--',tt,f_sum-(model.a+model.b*tt+model.c*tt.*tt),'r')
