/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/

#include "Variables.h"

// Function that catches the error 
void test(cudaError_t error, const char *file, int line)  {
	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

////////////////////////////////////////////////////////////////
// Memory for RNG use 
////////////////////////////////////////////////////////////////
// The state variables of CMRG on GPU 
TabSeedCMRG_t *pt_CMRG;
// The combination of CMRGs, the used RNGs
Tab2RNG_t *pt_2RNG;
// The state variables of CMRG on CPU
TabSeedCMRG_t *pt_CMRGCPU;
// Matrices associated to the post treatment of the CMRG
// - First MRG
double A1[3][3];
// - Second MRG
double A2[3][3];

float *m11;
float *m12;
float *m13;
float *m21;
float *m22;
float *m23;

float *X;
float *XI;
float *Y;
float *Z;
float *SIG;
float *Alpha;
float *Matcorr;
float *Matcorr2;
float *GammaY;

float *Cst;
float *GammaZ;
float *GamGPUY;
float *CstGPU;
float *GamGPUZ;

float *val1;
float *var1;
float *val2;
float *var2;

float *EZ;
float *VZ;
float *EZt;
float *VZt;
float *EY;
float *VY;
float *EYt;
float *VYt;