#include <stdio.h>
#include <math.h>
#include <complex.h>

void SolveCubicEquation(double complex x[],double a,double b,double c,double d);
void SolveQuarticEquation(double complex x[],double a,double b,double c,double d,double e);
double Cuberoot(double x);
double complex Squreroot(double complex x);
/*
int main(void){
    double g0=2;
    double alpha=5;
    double beta=10;
    double gamma=0.2;
    double Topt=28;

    double complex y[4];

    double a;
    double b;
    double c;
    double d;
    double e;

    double Tf=34.7;

    
    for(int j=0;j<4000;j++){
        y[0]=0;
        y[1]=0;
        y[2]=0;
        y[3]=0;
        Tf=j*0.01;

        a=alpha*alpha*g0/beta/beta;
        b=0-3*alpha*alpha*g0/beta/beta-2*alpha*g0*(Tf-Topt)/beta/beta;
        c=3*alpha*alpha*g0/beta/beta+4*alpha*g0*(Tf-Topt)/beta/beta-g0+(Tf-Topt)*(Tf-Topt)*g0/beta/beta;
        d=0-alpha*alpha*g0/beta/beta-2*alpha*(Tf-Topt)*g0/beta/beta-(Tf-Topt)*(Tf-Topt)*g0/beta/beta+g0-gamma;
        e=0;

        SolveQuarticEquation(y,a,b,c,d,e);//Solve 4D equations
        for(int i=0;i<4;i++){
            if(cimag(y[i])==0){
                printf("%lf,%lf\n",Tf,creal(y[i]));
            }
        }
    }
    

    for(int j=0;j<100;j++){
        y[0]=0;
        y[1]=0;
        y[2]=0;
        y[3]=0;

        Topt=j*0.1+25;

        for(int k=0;k<100;k++){
            g0=k*0.04;
            a=alpha*alpha*g0/beta/beta;
            b=0-3*alpha*alpha*g0/beta/beta-2*alpha*g0*(Tf-Topt)/beta/beta;
            c=3*alpha*alpha*g0/beta/beta+4*alpha*g0*(Tf-Topt)/beta/beta-g0+(Tf-Topt)*(Tf-Topt)*g0/beta/beta;
            d=0-alpha*alpha*g0/beta/beta-2*alpha*(Tf-Topt)*g0/beta/beta-(Tf-Topt)*(Tf-Topt)*g0/beta/beta+g0-gamma;
            e=0;      

            SolveQuarticEquation(y,a,b,c,d,e);
            if((cimag(y[3])==0)&(creal(y[3])>0.1)){
                printf("%lf,%lf,%lf\n",g0,Topt,creal(y[3]));
            }    
        }
    }
    
}
*/
int main(){
    double a;
    double b;
    double c;
    double d;
    double mu2=6.2;
    
    double Tref=25;
    double Tth=29;

    double Fth=1.3;
    double Fref=1;

    double complex x[3];
    
    for(int j=0;j<4000;j++){
        x[0]=0;
        x[1]=0;
        x[2]=0;
        double Tf=j*0.01;
        double F=Fref+(Tf-Tref)/(Tth-Tref)*(Fth-Fref);
        

        a=mu2;
        b=(-1)*2*mu2;
        c=mu2+1;
        d=(-1)*F;

        SolveCubicEquation(x,a,b,c,d);//Solve 4D equations
        for(int i=0;i<3;i++){
            if(cimag(x[i])==0){
                printf("%lf,%lf\n",F,creal(x[i]));
            }
        }

    }
}

void SolveCubicEquation(double complex x[],double a,double b,double c,double d){
    double A,B,C,p,q,D;
    A = b/a;
    B = c/a;
    C = d/a;
    p = B-A*A/3.0;
    q = 2.0*A*A*A/27.0-A*B/3.0+C;
    D = q*q/4.0+p*p*p/27.0;
    if(D < 0.0){//three real solutions
        double theta = atan2(sqrt(-D),-q*0.5);
        x[0] = 2.0*sqrt(-p/3.0)*cos(theta/3.0)-A/3.0;
        x[1] = 2.0*sqrt(-p/3.0)*cos((theta+2.0*M_PI)/3.0)-A/3.0;
        x[2] = 2.0*sqrt(-p/3.0)*cos((theta+4.0*M_PI)/3.0)-A/3.0;
    }
    else{//single real solution and two imaginary solutions(c.c)
        double u = Cuberoot(-q*0.5+sqrt(D)),v = Cuberoot(-q*0.5-sqrt(D));
        x[0] = u+v-A/3.0;
        x[1] = -0.5*(u+v)+sqrt(3.0)*0.5*1i*(u-v)-A/3.0;
        x[2] = -0.5*(u+v)-sqrt(3.0)*0.5*1i*(u-v)-A/3.0;
    }
}

void SolveQuarticEquation(double complex x[],double a,double b,double c,double d,double e){
    double A,B,C,D,p,q,r,epsilon_temp = 0.000001;
    A = b/a;
    B = c/a;
    C = d/a;
    D = e/a;
    p = -6.0*pow(A/4.0,2.0)+B;
    q = 8.0*pow(A/4.0,3.0)-2.0*B*A/4.0+C;
    r = -3.0*pow(A/4.0,4.0)+B*pow(A/4.0,2.0)-C*A/4.0+D;
    double complex t_temp[3];
    SolveCubicEquation(t_temp,1.0,-p,-4.0*r,4.0*p*r-q*q);
    double t = creal(t_temp[0]);
    double complex m = Squreroot(t-p);
    x[0] = (-m+Squreroot(-t-p+2.0*q/m))*0.5-A/4.0;
    x[1] = (-m-Squreroot(-t-p+2.0*q/m))*0.5-A/4.0;
    x[2] = (m+Squreroot(-t-p-2.0*q/m))*0.5-A/4.0;
    x[3] = (m-Squreroot(-t-p-2.0*q/m))*0.5-A/4.0;

}



double Cuberoot(double x){
    if(x > 0.0){
        return pow(x,1.0/3.0);
    }else{
        return -pow(-x,1.0/3.0);
    }
}

double complex Squreroot(double complex x){
    double complex y;
    double r = sqrt(creal(x)*creal(x)+cimag(x)*cimag(x)),theta = atan2(cimag(x),creal(x));
    if(cimag(x) == 0.0){
        if(creal(x) > 0.0){
            y = sqrt(r);
        }else{
            y = sqrt(r)*1i;
        }
    }else{
        if(theta < 0.0){
            theta += 2.0*M_PI;
        }
        double complex y = sqrt(r)*(cos(theta*0.5)+1i*sin(theta*0.5));
    }
    return y;
}
