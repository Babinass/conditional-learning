/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/

#include<stdio.h>
#include<stdlib.h>
#include <math.h>
#include "RNG.h"
#include "Functions.h"
#include "Variables.h"

///////////////////////////////////////////////////////
/////////////Main routine of the algorithm/////////////
///////////////////////////////////////////////////////
void CondMCLearn_BSDE(AP apB, AP apQ, AP apA, AP apH,  int flag){

	//Lauch the Conditional MC learning procedure
	if(flag == 0){//Bergam BSDE
		printf("\n Conditional MC learning procedure for Bergam BSDE \n");
		//Diffusion Parameters (rate, drift, coefdiff, init, time horizon)
		DP DPB = {0.06f, 0.06f, 0.2f, 100.0f, 0.5f};
		printf("The default parameters (rate, drift, coefdiff, init, time horizon, time discretization) are:\n");
		printf("r=%f,  mu =%f, sigma = %f, X0 =%f, T = %f, N=%i ", 0.06,0.06, 0.2, 100.0, 0.5, apB.NI);
    	float dtt = sqrt(DPB.T/ (float)apB.NI);
		CondMCLearn_BSDE_Bergam(apB, DPB, dtt, flag);
	}

	if(flag == 1){ //Quadra BSDE
		printf("\n Conditional MC learning procedure for Quadra BSDE \n");
		//Diffusion Parameters (rate, drift, coefdiff, init, time horizon)
		DP DPQ = {0.0f, 0.0f, 1.0f, 0.0f, 1.0f};
		printf("The default parameters (rate, drift, coefdiff, init, time horizon, time discretization) are:\n");
		printf("r=%f,  mu =%f, sigma = %f, X0 =%f, T = %f, N=%i ", 0.0, 0.0, 1.0, 0.0, 1.0, apQ.NI);		
    	float dtt = sqrt(DPQ.T/ (float)apQ.NI);		
		CondMCLearn_BSDE_Quadra(apQ, DPQ, dtt, flag);
	}

	if(flag == 2){//AllenCahn BSDE
		printf("\n Conditional MC learning procedure for AllenCahn BSDE \n");
		//Diffusion Parameters (rate, drift, coefdiff, init, time horizon)
		DP DPA = {0.0f, 0.0f, 1.4142f, 0.0f, 0.3f};
		printf("The default parameters (rate, drift, coefdiff, init, time horizon, time discretization) are:\n");
		printf("r=%f,  mu =%f, sigma = %f, X0 =%f, T = %f, N=%i ", 0.0f, 0.0f, 1.4142f, 0.0f, 0.3f, apA.NI);		
		float dtt = sqrt(DPA.T/ (float)apA.NI);
		CondMCLearn_BSDE_AllenCahn(apA, DPA, dtt, flag); 
	}

	if(flag == 3){//HJB BSDE
		printf("\n Conditional MC learning procedure for HJB BSDE \n");
		//Diffusion Parameters (rate, drift, coefdiff, init, time horizon)
		DP DPH = {0.0f, 0.0f, 1.4142f, 0.0f, 1.0f};
		printf("The default parameters (rate, drift, coefdiff, init, time horizon, time discretization) are:\n");
		printf("r=%f,  mu =%f, sigma = %f, X0 =%f, T = %f, N=%i ", 0.0f, 0.0f, 1.4142f, 0.0f, 1.0f, apA.NI);
    	float dtt = sqrt(DPH.T/ (float)apH.NI);
		CondMCLearn_BSDE_HJB(apH, DPH, dtt, flag); 
	}		

	//Output of the algorithm
	float  valCPU, varCPU;
	testCUDA(cudaMemcpy(&valCPU, val1, sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(&varCPU, var1, sizeof(float), cudaMemcpyDeviceToHost));
	printf("\n\n (Y0 lear = %f, std = %f) \n", valCPU,
	sqrtf(varCPU - valCPU*valCPU)/sqrt((float)NbOuter));					

	testCUDA(cudaMemcpy(&valCPU, val2, sizeof(float), cudaMemcpyDeviceToHost));
	testCUDA(cudaMemcpy(&varCPU, var2, sizeof(float), cudaMemcpyDeviceToHost));
	printf(" (Y0 sim = %f, std = %f) \n", valCPU,
	sqrtf(varCPU - valCPU*valCPU)/sqrt((float)NbOuter));

}
/////////////////////////////////////////////////////////

int main(int argc, char *argv[]){
    testCUDA(cudaSetDeviceFlags(cudaDeviceMapHost));    
    float TimerV;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int flag;
	printf("The user has to choose between the following numbers, each one corresponds to an example of BSDE:\n");
	printf("0) Bergam BSDE\n");
	printf("1) Quadratic BSDE\n");
	printf("2) Allen-Cahn BSDE\n");
	printf("3) HJB BSDE\n");
	printf("Choose one of the above: ");
	scanf("%i", &flag);
    cudaEventRecord(start, 0);
	
	//Algorithm parameters (time discretization)
	AP APB = {4};
	AP APQ = {256};
	AP APA = {16};
	AP APH = {8};
	////////Memories allocation CPU and GPU/////////////
	if(flag == 0){
		RngMalloc();
		OutMalloc(APB);
		RegMalloc(APB);
		InMalloc(APB);
	}
	if(flag == 1){
		RngMalloc();
		OutMalloc(APQ);
		RegMalloc(APQ);
		InMalloc(APQ);		
	}
	if(flag == 2){
		RngMalloc();
		OutMalloc(APA);
		RegMalloc(APA);
		InMalloc(APA);		
	}
	if(flag == 3){
		RngMalloc();
		OutMalloc(APH);
		RegMalloc(APH);
		InMalloc(APH);		
	}		
	///////////////////////////////////////////////////


    PostInitDataCMRG();            //Initialisation of random number
	CondMCLearn_BSDE(APB, APQ, APA, APH, flag);     //Main routine of the algorithm


    ////////Memories deallocation CPU and GPU/////////////
	RngFree();
	OutFree();
	RegFree();
	InFree();
	///////////////////////////////////////////////////

    cudaEventRecord(stop, 0);	
    cudaEventSynchronize(stop);
	cudaEventElapsedTime(&TimerV,      //GPU timer instructions
		start, stop);
	printf("\n Execution time: %f s\n", TimerV/1000.0f); //GPU timer instructions
        
    return 0;
}