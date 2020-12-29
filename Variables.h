/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include "Parameter.h"

// Function that catches the error 
void test(cudaError_t error, const char *file, int line);

// Has to be defined in the compilation in order to get the correct value 
// of the macros __FILE__ and __LINE__
#define testCUDA(error) (test(error, __FILE__ , __LINE__))
#define SQR(A) (A*A)

////////////////////////////////////////////////////////////////
// Memory for RNG use 
////////////////////////////////////////////////////////////////
// The state variables of CMRG on GPU 
extern TabSeedCMRG_t *pt_CMRG;
// The combination of CMRGs, the used RNGs
extern Tab2RNG_t *pt_2RNG;
// The state variables of CMRG on CPU
extern TabSeedCMRG_t *pt_CMRGCPU;
// Matrices associated to the post treatment of the CMRG
// - First MRG
extern double A1[3][3];
// - Second MRG
extern double A2[3][3];


extern float *m11;
extern float *m12;
extern float *m13;
extern float *m21;
extern float *m22;
extern float *m23;

extern float *X;
extern float *XI;
extern float *Y;
extern float *Z;
extern float *SIG;
extern float *Alpha;
extern float *Matcorr;
extern float *Matcorr2;
extern float *GammaY;
extern float *Cst;

extern float *GammaZ;
extern float *GamGPUY;
extern float *CstGPU;
extern float *GamGPUZ;

extern float *val1;
extern float *var1;
extern float *val2;
extern float *var2;

extern float *EZ;
extern float *VZ;
extern float *EZt;
extern float *VZt;
extern float *EY;
extern float *VY;
extern float *EYt;
extern float *VYt;