/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/

#include "Variables.h"
using namespace std;

////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////
// Memory allocation for RNG 
////////////////////////////////////////////////////////////
void RngMalloc(void)
{
	testCUDA(cudaMalloc(&pt_CMRG, sizeof(TabSeedCMRG_t)));
	testCUDA(cudaMalloc(&pt_2RNG, sizeof(Tab2RNG_t)));
	testCUDA(cudaHostAlloc(&pt_CMRGCPU, sizeof(TabSeedCMRG_t),
		cudaHostAllocDefault));
}

////////////////////////////////////////////////////////////
// Memory deallocation for RNG 
////////////////////////////////////////////////////////////
void RngFree(void)
{
	testCUDA(cudaFree(pt_CMRG));
	testCUDA(cudaFree(pt_2RNG));
	testCUDA(cudaFreeHost(pt_CMRGCPU));
}

////////////////////////////////////////////////////////////
// Memory allocation for outer trajectories
////////////////////////////////////////////////////////////
void OutMalloc(AP ap)
{
	int NI = ap.NI;
	testCUDA(cudaMalloc(&m11, 2*NbOuter * sizeof(float)));
	testCUDA(cudaMalloc(&m12, 2*NbOuter * sizeof(float)));

	testCUDA(cudaMalloc(&m21, NbOuter * sizeof(float)));
	testCUDA(cudaMalloc(&m22, NbOuter * sizeof(float)));

	testCUDA(cudaMalloc(&SIG, 8 * sizeof(float)));
	testCUDA(cudaMalloc(&Alpha, NbOuter * sizeof(float)));

	testCUDA(cudaMalloc(&X, (NI + 1)*Dim*NbOuter*sizeof(float)));

	testCUDA(cudaMalloc(&Y, (NI+1)*NbOuter*sizeof(float)));
	testCUDA(cudaMalloc(&Z, NI*Dim*NbOuter*sizeof(float)));

	testCUDA(cudaMalloc(&val1, sizeof(float)));
	testCUDA(cudaMalloc(&var1, sizeof(float)));
	testCUDA(cudaMalloc(&val2, sizeof(float)));
	testCUDA(cudaMalloc(&var2, sizeof(float)));		

	testCUDA(cudaMalloc(&EZ, NI*Dim*sizeof(float)));
	testCUDA(cudaMalloc(&VZ, NI*Dim*sizeof(float)));
	testCUDA(cudaMalloc(&EZt, NI*Dim*sizeof(float)));
	testCUDA(cudaMalloc(&VZt, NI*Dim*sizeof(float)));	
	testCUDA(cudaMalloc(&EY, NI*sizeof(float)));
	testCUDA(cudaMalloc(&VY, NI*sizeof(float)));	
	testCUDA(cudaMalloc(&EYt, NI*sizeof(float)));
	testCUDA(cudaMalloc(&VYt, NI*sizeof(float)));		
}

////////////////////////////////////////////////////////////
// Memory deallocation for outer trajectories
////////////////////////////////////////////////////////////
void OutFree(void)
{
	testCUDA(cudaFree(m11));
	testCUDA(cudaFree(m12));

	testCUDA(cudaFree(m21));
	testCUDA(cudaFree(m22));

	testCUDA(cudaFree(SIG));

	testCUDA(cudaFree(Alpha));

	testCUDA(cudaFree(X));
	testCUDA(cudaFree(Y));
	testCUDA(cudaFree(Z));	

	testCUDA(cudaFree(val1));
	testCUDA(cudaFree(var1));	
	testCUDA(cudaFree(val2));
	testCUDA(cudaFree(var2));	

	testCUDA(cudaFree(EZ));
	testCUDA(cudaFree(VZ));	
	testCUDA(cudaFree(EZt));
	testCUDA(cudaFree(VZt));	
	testCUDA(cudaFree(EY));
	testCUDA(cudaFree(VY));
	testCUDA(cudaFree(EYt));
	testCUDA(cudaFree(VYt));		
}


////////////////////////////////////////////////////////////
// Memory allocation for regressed values
////////////////////////////////////////////////////////////
void RegMalloc(AP ap)
{
	int NI = ap.NI;
	testCUDA(cudaMalloc(&Matcorr, NbOuter*Dim*Dim * sizeof(float)));
	
	testCUDA(cudaHostAlloc(&GammaZ, NbOuter*Dim*Dim*sizeof(float), cudaHostAllocMapped));
	testCUDA(cudaHostAlloc(&GammaY, ((NI*(NI-1))/2)*NbOuter*Dim *sizeof(float), cudaHostAllocMapped));
	testCUDA(cudaHostAlloc(&Cst, ((NI*(NI - 1))/2)*NbOuter * sizeof(float), cudaHostAllocMapped));
}

////////////////////////////////////////////////////////////
// Memory deallocation for regressed values
////////////////////////////////////////////////////////////
void RegFree(void)
{
	testCUDA(cudaFree(Matcorr));
	testCUDA(cudaFreeHost(GammaZ));
	testCUDA(cudaFreeHost(GammaY));	
	testCUDA(cudaFreeHost(Cst));
}

////////////////////////////////////////////////////////////
// Memory allocation for inner trajectories
////////////////////////////////////////////////////////////
void InMalloc(AP ap)
{
	testCUDA(cudaMalloc(&XI, 2*Dim*NbOuter*NbInner * sizeof(float)));
}

////////////////////////////////////////////////////////////
// Memory deallocation for inner trajectories
////////////////////////////////////////////////////////////
void InFree(void)
{
	testCUDA(cudaFree(XI));
}