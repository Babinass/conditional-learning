/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "Variables.h"
#include "Parameter.h"

void VarMalloc(AP ap);
void VarFree(void);

void RngMalloc(void);
void RngFree(void);

void OutMalloc(AP ap);
void OutFree(void);

void RegMalloc(AP ap);
void RegFree(void);

void InMalloc(AP ap);
void InFree(void);

void CondMCLearn_BSDE_Bergam(AP APB, DP DPB, float dtt, int flag);
void CondMCLearn_BSDE_Quadra(AP APQ, DP DPQ, float dtt, int flag);
void CondMCLearn_BSDE_AllenCahn(AP APA, DP DPA, float dtt, int flag);
void CondMCLearn_BSDE_HJB(AP APH, DP DPH, float dtt, int flag);