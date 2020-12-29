/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/
#ifndef __CONSTANTS_TYPES__
#define __CONSTANTS_TYPES__

////////////////Algorithm Parameters///////////////
//The user must choose right parameters for each BSDE. 
//This following options are suited to the default
//model and diffusion parameters.

//Number of MC trajectories for outer simulation
//For Bergam BSDE:  NbOuter (128)
//For Quadra BSDE:  NbOuter (128)
//For AllenCahn BSDE:  NbOuter (16)
//For HJB BSDE:  NbOuter (128)
#define NbOuter (16)          

//Number of MC trajectories for inner simulation
//For Bergam BSDE:  NbOuter (2048)
//For Quadra BSDE:  NbOuter (128)
//For AllenCahn BSDE:  NbOuter (64)
//For HJB BSDE:  NbOuter (256*128)
#define NbInner (64)               

//Increase (sequentially) inner trajectories
//For Bergam BSDE:  NbOuter (1)
//For Quadra BSDE:  NbOuter (1)
//For AllenCahn BSDE:  NbOuter (1)
//For HJB BSDE:  NbOuter (4)
#define NbInTimes (1)  

//Dimension of the problem
#define Dim (100)                   

//Number of coarse time step 
typedef struct Algo_Params AP;
struct Algo_Params{
    int NI;
};
////////////////////////////////////////////////////

///////////// Management of block and thread/////////
#define TotInner (NbInner*NbInTimes)      //Total number of inner trajectories 
#define NM ((NbInner) < (256) ? (32):(8)) //Increase number of inner trajectories for matrix regression
#define NbMat (NbInner*NM)                //Total number of inner trajectories for matrix regression
#define Total (NbOuter*NbInner)           //Total number of trajectories

#define BlockOuter ( (NbOuter) < 256 ? (NbOuter):(256))
#define GridOuter (NbOuter/BlockOuter)
#define MinNbT 256
#define GridInnerY ( (NbInner) < (MinNbT) ? (1):(NbInner/MinNbT))
#define BlockInnerX ( (NbInner) < (MinNbT) ? (NbInner):(MinNbT))
#define GridInnerX ( (NbInner) < (MinNbT) ? (Total/MinNbT):(NbOuter) )
#define BlockInnerY (NbOuter/GridInnerX)
#define BlockInner (128)
///////////////////////////////////////////////////////


//////////////General model parameters/////////////////
typedef struct DiffuParams DP;
struct DiffuParams{
    float r;      //rate diffusion
    float mu;     //drift diffusion
    float sigma;  //coefficien of diffusion
    float xi;     //initialisation of the diffusion
    float T;      //Time horizon  
};

///////////////////////////////////////////////////////

///////////Specific model parameters///////////////////
/*******Bergam BSDE***********/
#define Rl (4.0f/100.0f)
#define Rb (6.0f/100.0f) 
/****************************/

/*******Quadra BSDE***********/
#define alph (0.4f)
/****************************/

/*******Allen Cahn BSDE***********/
//Nothing
/****************************/

/*******HJB BSDE***********/

/****************************/

////////////////////////////////////////////////////////
// Pi approximation needed in some kernels
#define MoPI (3.1415927f)

////////////////////////////////////////////////////////////////
// L'Eucuyer CMRG Matrix Values
////////////////////////////////////////////////////////////////
// First MRG
#define a12 63308
#define a13 -183326
#define q12 33921
#define q13 11714
#define r12 12979
#define r13 2883

// Second MRG
#define a21 86098
#define a23 -539608
#define q21 24919
#define q23 3976
#define r21 7417
#define r23 2071

// Normalization variables
#define Invmp 4.6566129e-10f
#define two17 131072.0
#define two53 9007199254740992.0

typedef int TabSeedCMRG_t[NbInner][NbOuter][6];
typedef float Tab2RNG_t[NbInner][NbOuter][2];

#endif