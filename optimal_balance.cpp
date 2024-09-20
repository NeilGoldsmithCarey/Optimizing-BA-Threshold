//This program calculates optimal data balance, regression lines, and balanced accuracy
//for threshold models involving bivariate normal densities, as described in the paper
//"Optimizing Balanced Accuracy in Medical Data Threshold Models"
//Authors: Benjamin F. Dribus, Jamie Hill, Neil Goldsmith, Daniel Martingano
//
//Choice of density is given by specifying a slope m and threshold L in the main function

//PACKAGES
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <numeric>
#include <random>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <functional>
#include <thread>

//METHODS
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////


//gaussian tail using trapezoidal integration//////////////////////////////////////////////
float gauss_tail(float argument){
    
    float xmin=argument;
    float xmax=10+argument;
    int xsubd=20000;

    float dx=(xmax-xmin)/xsubd;

    float working_sum=0;
    float left;
    float right;
    float x;
    for(int i=0; i<xsubd; i++){
        x=xmin+i*dx;
        left=std::exp(-x*x);
        right=std::exp(-(x+dx)*(x+dx));
        working_sum+=(left+right)*dx/2;
    }

    return (2/sqrt(3.14159265359))*working_sum;
}

//erf using gaussian tail//////////////////////////////////////////////////////////////////
float erf(float argument){

    if(argument>=0){
        return 1-gauss_tail(argument);
    }
    else{
        return gauss_tail(-argument)-1;
    }     
}

//erfc using gaussian tail/////////////////////////////////////////////////////////////////
float erfc(float argument){

    if(argument>=0){
        return gauss_tail(argument);
    }
    else{
        return 2-gauss_tail(-argument);
    }     
}

//normalization constants//////////////////////////////////////////////////////////////////
float c_plus(float m,float L){
    return 2/erfc(L/sqrt(1+m*m));
}
float c_minus(float m,float L){
    return 2/erfc(-L/sqrt(1+m*m));
}

//Expressions in Theorem 1/////////////////////////////////////////////////////////////////
float E_plus(float m,float L){
    return (1/(sqrt(3.14159265359)*sqrt(1+m*m)))*std::exp(-L*L/(1+m*m))/erfc(L/sqrt(1+m*m));
}

float E_minus(float m,float L){
    return (1/(sqrt(3.14159265359)*sqrt(1+m*m)))*std::exp(-L*L/(1+m*m))/erfc(-L/sqrt(1+m*m));
}
float F_m(float m){
    return (1+m*m)/(2*(sqrt(1+m*m)-1));
}
float G_m(float m){
    return ((1+m*m)*(2*sqrt(1+m*m)-1))/((sqrt(1+m*m)-1));
}
//Regression line expressions//////////////////////////////////////////////////////////////
float A_plus(float m, float L){
    return .5+((m*m)/(1+m*m))*L*E_plus(m,L);
}
float A_minus(float m, float L){
    return .5-(m*m/(1+m*m))*L*E_minus(m,L);
}
float B_plus(float m, float L){
    return m*E_plus(m,L);
}
float B_minus(float m, float L){
    return -m*E_minus(m,L);
}
float C_plus(float m, float L){
    return .5*m+m*L*E_plus(m,L);
}
float C_minus(float m, float L){
    return .5*m-m*L*E_minus(m,L);
}
float D_plus(float m, float L){
    return (1+m*m)*E_plus(m,L);
}
float D_minus(float m, float L){
    return -(1+m*m)*E_minus(m,L);
}
float A_lambda(float lambda, float m, float L){
    return lambda*A_plus(m,L)+(1-lambda)*A_minus(m,L);
}
float B_lambda(float lambda, float m, float L){
    return lambda*B_plus(m,L)+(1-lambda)*B_minus(m,L);
}
float C_lambda(float lambda, float m, float L){
    return lambda*C_plus(m,L)+(1-lambda)*C_minus(m,L);
}
float D_lambda(float lambda, float m, float L){
    return lambda*D_plus(m,L)+(1-lambda)*D_minus(m,L);
}
float m_lambda(float lambda, float m, float L){
    return (C_lambda(lambda,m,L)-B_lambda(lambda,m,L)*D_lambda(lambda,m,L))/(A_lambda(lambda,m,L)-B_lambda(lambda,m,L)*B_lambda(lambda,m,L));
}
float b_lambda(float lambda, float m, float L){
    return (A_lambda(lambda,m,L)*D_lambda(lambda,m,L)-B_lambda(lambda,m,L)*C_lambda(lambda,m,L))/(A_lambda(lambda,m,L)-B_lambda(lambda,m,L)*B_lambda(lambda,m,L));
}
float x_lambda(float lambda, float m, float L){
    float working_m=m_lambda(lambda,m,L);
    float working_b=b_lambda(lambda,m,L);

    return (L-working_b)/working_m;

}
//Optimal balance////////////////////////////////////////////////////////////////////////
float lambda_opt(float m,float L){
    float mu=sqrt(1+m*m);
    float F=mu*mu/(2*(mu-1));
    float G=mu*mu*(2*mu-1)/(mu-1);
    float num1=L*L+2*mu*mu*L*E_minus(m,L)-F;
    float num2=sqrt(L*L*L*L+G*L*L+F*F);
    float denom=2*mu*mu*L*(E_plus(m,L)+E_minus(m,L));

    return (num1+num2)/denom;
}
float lambda_minus(float m,float L){
    float mu=sqrt(1+m*m);
    float F=mu*mu/(2*(mu-1));
    float G=mu*mu*(2*mu-1)/(mu-1);
    float num1=L*L+2*mu*mu*L*E_minus(m,L)-F;
    float num2=sqrt(L*L*L*L+G*L*L+F*F);
    float denom=2*mu*mu*L*(E_plus(m,L)+E_minus(m,L));

    return (num1-num2)/denom;
}
float x_opt(float m,float L){
    return (L/m)*(1-1/sqrt(1+m*m));
}
float x_actual(float L,float m,float b){
    return (L-b)/m;
}

//actual positive accuracy for bivariate normal density///////////////////////////////////
float actual_PA(float L, float m, float x_in){

    float xmin=x_in;
    float xmax=10;
    float ymin=L;
    float ymax=10;
    int xsubd=5000;
    int ysubd=5000;
    float dx=(xmax-xmin)/xsubd;
    float dy=(ymax-ymin)/ysubd;
    float dA=dx*dy;
    float working_sum=0;
    float h;
    float x;
    float y;
    for(int i=0; i<xsubd; i++){
        for(int j=0; j<ysubd; j++){
            x=xmin+i*dx;
            y=ymin+j*dy;
            h=std::exp(-x*x)*std::exp(-(y-m*x)*(y-m*x));
            working_sum+=h*dA;
        }
    }
    return working_sum*c_plus(m,L)/3.14159265359;
}
float PA_lambda(float L, float m, float lambda){
    return actual_PA(L,m,x_lambda(lambda,m,L));
}

//actual negative accuracy for bivariate normal density////////////////////////////////////
float actual_NA(float L, float m, float x_in){
    float xmin=-10;
    float xmax=x_in;
    float ymin=-10;
    float ymax=L;
    int xsubd=5000;
    int ysubd=5000;
    float dx=(xmax-xmin)/xsubd;
    float dy=(ymax-ymin)/ysubd;
    float dA=dx*dy;
    float working_sum=0;
    float h;
    float x;
    float y;
    for(int i=0; i<xsubd; i++){
        for(int j=0; j<ysubd; j++){
            x=xmin+i*dx;
            y=ymin+j*dy;
            h=std::exp(-x*x)*std::exp(-(y-m*x)*(y-m*x));
            working_sum+=h*dA;
        }
    }
    return working_sum*c_minus(m,L)/3.14159265359;
}
float NA_lambda(float L, float m, float lambda){
    return actual_NA(L,m,x_lambda(lambda,m,L));
}

//actual balanced accuracy for bivariate normal density///////////////////////////////////////
float actual_BA(float L, float m, float x_in){
    return .5*(actual_PA(L,m,x_in)+actual_NA(L,m,x_in));
}

float BA_lambda(float L, float m, float lambda){
    return actual_BA(L,m,x_lambda(lambda,m,L));
}

//negative proportion of data//////////////////////////////////////////////////////////////////
float neg_prop(float L, float m){
    float pos_weight;
    float neg_weight;
    float xmin=-10;
    float xmax=10;
    float ymin=L;
    float ymax=10;
    int xsubd=5000;
    int ysubd=5000;
    float dx=(xmax-xmin)/xsubd;
    float dy=(ymax-ymin)/ysubd;
    float dA=dx*dy;
    float working_sum=0;
    float h;
    float x;
    float y;
    for(int i=0; i<xsubd; i++){
        for(int j=0; j<ysubd; j++){
            x=xmin+i*dx;
            y=ymin+j*dy;
            h=std::exp(-x*x)*std::exp(-(y-m*x)*(y-m*x));
            working_sum+=h*dA;
        }
    }

    pos_weight=working_sum;

    ymin=-10;
    ymax=L;
    dx=(xmax-xmin)/xsubd;
    dy=(ymax-ymin)/ysubd;
    dA=dx*dy;
    working_sum=0;
    for(int i=0; i<xsubd; i++){
        for(int j=0; j<ysubd; j++){
            x=xmin+i*dx;
            y=ymin+j*dy;
            h=std::exp(-x*x)*std::exp(-(y-m*x)*(y-m*x));
            working_sum+=h*dA;
        }
    }

    neg_weight=working_sum;

    return neg_weight/(neg_weight+pos_weight);

}


//MAIN
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////

int main(void){


float m=1;
float L=2;
float main_lambda=lambda_opt(m,L);
float main_slope=m_lambda(main_lambda,m,L);
float main_intercept=b_lambda(main_lambda,m,L);
float main_x=x_actual(L,main_slope,main_intercept);
float main_x_opt=x_opt(m,L);
float main_BA=actual_BA(L,m,main_x);
float inc=.01;
float main_x_minus=main_x_opt-inc;
float main_x_plus=main_x_opt+inc;
float BA_left=actual_BA(L,m,main_x_minus);
float BA_right=actual_BA(L,m,main_x_plus);

std::cout << "if m=" << m <<  " and L=" << L << " then the optimal proportion of positive data (by Theorem 1) is " << main_lambda << std::endl;
std::cout << "this gives a least-squares regression line with slope " << main_slope <<  " and intercept " << main_intercept << std::endl;
std::cout << "this line intersects the threshold y=" << L <<  " at x=" << main_x << std::endl;
std::cout << "the balanced accuracy via numerical integration for this intersection is BA=" << main_BA << std::endl;
std::cout << "moving to the left by " << inc << ", BA=" << BA_left << std::endl;
std::cout << "moving to the right by " << inc << ", BA=" << BA_right << std::endl;
std::cout << "the theoretical best intersection (by Lemma 3.3.1) is x=" << main_x_opt << std::endl;


return 0;

}