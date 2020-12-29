/**********************************************************************
Code associated to the paper:

CONDITIONAL MONTE CARLO LEARNING FOR DIFFUSIONS

by: 

Lokman A. Abbas-Turki, Babacar Diallo and Giles Pagès

Those who re-use this code should mention in their code 
the name of the authors above.
**********************************************************************/

#include "Functions.h"
#include "RNG.h"
#include "Variables.h"

/***************************************************************
* * *   *****  *       *  *   *****   *****      ******
*    *  *       *     *   *  *        *          *
*    *  *****    *   *    *  *        *****      ****
*    *  *         * *     *  *        *          *
* * *   *****      *      *   *****   *****      *    *
***************************************************************/
// Set the new RNG seed
static __device__ void CMRG_set_d(int *a0, int *a1, int *a2, int *a3, int *a4,
	int *a5, int *pt_CMRG_Out){
	pt_CMRG_Out[0] = *a0;
	pt_CMRG_Out[1] = *a1;
	pt_CMRG_Out[2] = *a2;
	pt_CMRG_Out[3] = *a3;
	pt_CMRG_Out[4] = *a4;
	pt_CMRG_Out[5] = *a5;
}

// Get the RNG Seed
static __device__ void CMRG_get_d(int *a0, int *a1, int *a2, int *a3, int *a4,
	int *a5, int *pt_CMRG_In){
	*a0 = pt_CMRG_In[0];
	*a1 = pt_CMRG_In[1];
	*a2 = pt_CMRG_In[2];
	*a3 = pt_CMRG_In[3];
	*a4 = pt_CMRG_In[4];
	*a5 = pt_CMRG_In[5];
}

//Generated random uniformly number
static __device__ void CMRG_d(int *a0, int *a1, int *a2, int *a3, int *a4,
	int *a5, float *g0, float *g1, int nb){

	const int m1 = 2147483647;// Requested for the simulation
	const int m2 = 2145483479;// Requested for the simulation
	int h, p12, p13, p21, p23, k, loc;// Requested local parameters

	for (k = 0; k<nb; k++){
		// First Component 
		h = *a0 / q13;
		p13 = a13*(h*q13 - *a0) - h*r13;
		h = *a1 / q12;
		p12 = a12*(*a1 - h*q12) - h*r12;

		if (p13 < 0) {
			p13 = p13 + m1;
		}
		if (p12 < 0) {
			p12 = p12 + m1;
		}
		*a0 = *a1;
		*a1 = *a2;
		if ((p12 - p13) < 0){
			*a2 = p12 - p13 + m1;
		}
		else {
			*a2 = p12 - p13;
		}

		// Second Component 
		h = *a3 / q23;
		p23 = a23*(h*q23 - *a3) - h*r23;
		h = *a5 / q21;
		p21 = a21*(*a5 - h*q21) - h*r21;

		if (p23 < 0){
			p23 = p23 + m2;
		}
		if (p12 < 0){
			p21 = p21 + m2;
		}
		*a3 = *a4;
		*a4 = *a5;
		if ((p21 - p23) < 0) {
			*a5 = p21 - p23 + m2;
		}
		else {
			*a5 = p21 - p23;
		}

		// Combines the two MRGs
		if (*a2 < *a5){
			loc = *a2 - *a5 + m1;
		}
		else{ loc = *a2 - *a5; }

		if (k){
			if (loc == 0){
				*g1 = Invmp*m1;
			}
			else{ *g1 = Invmp*loc; }
		}
		else{
			*g1 = 0.0f;
			if (loc == 0){
				*g0 = Invmp*m1;
			}
			else{ *g0 = Invmp*loc; }
		}
	}
}

// Generates Gaussian distribution from a uniform one (Box-Muller)
static __device__ void BoxMuller_d(float *g0, float *g1){

	float loc;
	if (*g1 < 1.45e-6f){
		loc = sqrtf(-2.0f*logf(0.00001f))*cosf(*g0*2.0f*MoPI);
	}
	else {
		if (*g1 > 0.99999f){
			loc = 0.0f;
		}
		else { loc = sqrtf(-2.0f*logf(*g1))*cosf(*g0*2.0f*MoPI); }
	}
	*g0 = loc;
}

//Generates Geometric Brownian Motion
static __device__ void GBM_d(float *res, DP dp, float dt, float g0){
    float temp;
    temp = (*res)*expf((dp.r - 0.5f*dp.sigma*dp.sigma)*dt*dt + dp.sigma*dt*g0);
    *res = temp;   
}

static __device__ void Diff_d(float *res, DP dp, float dt, float g0){
    float temp;
    temp = (*res) + dp.sigma*dt*g0;
    *res = temp;   
}

//One-Dimensional Normal Law. Cumulative distribution function. 
__device__ float NP_d(float x)
{
	float p = 0.2316419f;
	float b1 = 0.3193815f;
	float b2 = -0.3565638f;
	float b3 = 1.781478f;
	float b4 = -1.821256f;
	float b5 = 1.330274f;
	float one_over_twopi = 0.3989423f;
	float t;

	if (x >= 0.0f) {
		t = 1.0f / (1.0f + p * x);
		return (1.0f - one_over_twopi * expf(-x * x / 2.0f) * t *
			(t *(t * (t * (t * b5 + b4) + b3) + b2) + b1));
	}
	else {/* x < 0 */
		t = 1.0f / (1.0f - p * x);
		return (one_over_twopi * expf(-x * x / 2.0f) * t *
			(t *(t * (t * (t * b5 + b4) + b3) + b2) + b1));
	}
}
/***************************************************************
*   *  *****  *****  *   *  *****  *      *****
*  *   *      *   *  **  *  *      *      *
***    *****  ****   * * *  *****  *      *****
*  *   *      *   *  *  **  *      *          *
*   *  *****  *   *  *   *  *****  *****  *****
***************************************************************/
//Lauch outer trajectories
__global__ void GeneratePathsOuter_k(float *X_in, float dt, TabSeedCMRG_t *pt_cmrg, AP ap, DP dp, int flag){
	int gb_index_x = threadIdx.x + blockIdx.x*blockDim.x; //Index
    int a0, a1, a2, a3, a4, a5;
	float g0, g1, res;
	int NI = ap.NI; 
    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][0][gb_index_x]);//Get the seed

 
	for(int d = 0; d < Dim; d++){
		X_in[gb_index_x + d*blockDim.x*gridDim.x + 0*NbOuter*Dim] = dp.xi;
    	res = dp.xi;
		for(int i = 1; i<=NI; i++){
			CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2); //Random uniform variable
			BoxMuller_d(&g0, &g1); //Gaussian variable (Box Muller) 
			if(flag == 0){//Bergam BSDE
				GBM_d(&res, dp, dt, g0);//Geometric Brownian Motion
			}
			if(flag == 1){//Quadra BSDE
				Diff_d(&res, dp, dt, g0); //Diffusion
			}
			if(flag == 2){//AllenCahn BSDE
				Diff_d(&res, dp, dt, g0); //Diffusion
			}							
			if(flag == 3){//HJB BSDE
				Diff_d(&res, dp, dt, g0); //Diffusion
			}				
            X_in[gb_index_x + d*NbOuter + i*NbOuter*Dim] =  res;
        }
    }
    
	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][0][gb_index_x]);//Set the seed
}

//Coarse and fine approximation kernel for Z (Bergam BSDE)
__global__ void MCZ_Bergam_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					  float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apB, DP dpB, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, ii, it, d, dd;
	float g0, g1, res, res1, dw;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	float sigma = dpB.sigma;
	float mu = dpB.mu;
	int NI = apB.NI;
    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (dd = 0; dd < Dim; dd++){					
				res = sR[dd + threadIdx.y*Dim];
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				BoxMuller_d(&g0, &g1);
				GBM_d(&res, dpB, dtt, g0);
				XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total] = res;
			}


			if(i == NI - 1){
				res1 = -10000.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 = fmaxf(res1, XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total]);
				}

				res1 = (fmaxf(res1 - 120.0f, 0.0f) - 2.0f*fmaxf(res1 - 150.0f, 0.0f));
			} else {
				if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}				
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter]; 
				}else {
					res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}				
			}	
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for(d = 0; d < Dim; d++){
				dw = (logf(XI_tmp[gb_index_x + d*Total + ((i+1)%2)*Dim*Total]/
					sR[d + threadIdx.y*Dim]) - (mu - 0.5f*sigma*sigma)*dtt*dtt)/sigma;
	
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (1.0f/(dtt*dtt))*res1*dw/((float)TotInner);
				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}
				if (threadIdx.x == 0){
					atomicAdd(Zt + n_idx + d*NbOuter + i*Dim*NbOuter, sC[threadIdx.y*BlockInnerX]);
				}
			}	
		}
	} else{
		/////////////////////////////////// Replaces Matcorr_k i.e. computes E(\Xi ^t\Xi)
		if (threadIdx.x == 0) {
			for (d = 0; d < Dim; d++) {
				sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				sR[d + threadIdx.y*Dim + Dim*BlockInnerY] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
			}
		}
		__syncthreads();
		
		for(it=0; it<NM; it++){
			for (ii = i; ii < ie-1; ii++) { // First ie is equal to NI
				for (d = 0; d < Dim; d++) {
					if (ii == i) {
						res = sR[d + threadIdx.y*Dim];
					}
					else {
						res = XI_tmp[gb_index_x + d*Total + (ii%2)*Dim*Total];
					}
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					GBM_d(&res, dpB, dtt, g0);
					XI_tmp[gb_index_x + d*Total + ((ii+1)%2)*Dim*Total] = res;
				}
			}

			for(d = 0; d < Dim; d++){
				for(dd = 0; dd <= d; dd++){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] = (XI_tmp[gb_index_x + d*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[d + threadIdx.y*Dim + Dim*BlockInnerY])
															*(XI_tmp[gb_index_x + dd*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[dd + threadIdx.y*Dim + Dim*BlockInnerY])/(NbMat);
					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}
					if (threadIdx.x == 0){
						atomicAdd(Mat + n_idx*Dim*Dim + dd*Dim + d, sC[threadIdx.y*BlockInnerX]);
						atomicAdd(Mat + n_idx*Dim*Dim + d*Dim + dd, sC[threadIdx.y*BlockInnerX]);
					}
				}
			}

		}

		////////////////////////////////////////////////// Computes E(\Phi \Xi)//////////////////
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed

		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0) {
				for (d = 0; d < Dim; d++) {
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (d = 0; d < Dim; d++) {
				res = sR[d + threadIdx.y*Dim];
				for (ii = i; ii < ie; ii++) { // First ie is equal to NI
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					GBM_d(&res, dpB, dtt, g0);
					XI_tmp[gb_index_x + d*Total	+ ((ii + 1) % 2)*Dim*Total] = res;
				}
			}

			for(d = 0; d < Dim; d++){
				dw = (logf(XI_tmp[gb_index_x + d*Total + (ie%2)*Dim*Total]/
					XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total]) -
					(mu - 0.5f*sigma*sigma)*dtt*dtt)/sigma;

				if(ie == NI){
					res1 = -10000.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 = fmaxf(res1, XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total]);
					}

					res = (1.0f/(dtt*dtt))*(fmaxf(res1 - 120.0f, 0.0f) - 2.0f*fmaxf(res1 - 150.0f, 0.0f))*dw; 					
				} else {
					if((ni==NI) || (ie<ni)){
						if(threadIdx.x == 0){
							for(dd = 0; dd < Dim; dd++){
								sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
							}
						}
						__syncthreads();

						res1 = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
									(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
						}	
						if(flag == 0){
							res1 += Yt[n_idx + ie*NbOuter]; 
						}else {
							res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
						}										

						res =  (1.0f/(dtt*dtt))*res1*dw;
					} else {
						if(threadIdx.x == 0){
							for(dd = 0; dd < Dim; dd++){
								sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
							}
						}
						__syncthreads();

						res1 = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
									(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
						}	
						if(flag == 0){
							res1 += Yt[n_idx + ie*NbOuter]; 
						}else {
							res1 +=  cst[n_idx + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter];
						}											

						res =  (1.0f/(dtt*dtt))*res1*dw;
					}
				}
				__syncthreads();

				if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();

				for(dd = 0; dd < Dim; dd++){ 
					gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res - Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter])*
																(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total]-
																				sR[dd + threadIdx.y*Dim])/((float)TotInner);

					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}		
					if(threadIdx.x == 0){
						atomicAdd(gamZ + dd + d*Dim + n_idx*Dim*Dim, gamsh[threadIdx.y*BlockInnerX]);
					}
				}	
			}			
		}

	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

}

//Coarse and fine approximation kernel for Y (Bergam BSDE)
__global__ void MCY_Bergam_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apB, DP dpB, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, it, d, dd;
	float res, res1, fy;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	float sigma = dpB.sigma;
	float mu = dpB.mu;
	int NI = apB.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {

			if(i == NI - 1){
				res1 = -10000.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 = fmaxf(res1, XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total]);
				}
				res1 = fmaxf(res1 - 120.0f, 0.0f) - 2.0f*fmaxf(res1 - 150.0f, 0.0f);

				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res += Zt[n_idx + dd*NbOuter + i*Dim*NbOuter];
				}

				fy = -Rl*res1 - ((mu-Rl)/sigma)*res + (Rb-Rl)*fmaxf(res/sigma - res1, 0.0f);
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner); 
			}else{
			 	if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}					
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter]; 
				}else {
					res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}

				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res += Zt[n_idx + dd*NbOuter + i*Dim*NbOuter];
				}

				fy = -Rl*res1 - ((mu-Rl)/sigma)*res + (Rb-Rl)*fmaxf(res/sigma - res1, 0.0f);

				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner);
			}	
			__syncthreads();
			k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
				}
				__syncthreads();
				k /= 2;
			}
			if (threadIdx.x == 0){
				atomicAdd(Yt + n_idx +  i*NbOuter, sC[threadIdx.y*BlockInnerX]);
			}	
		}
	} else{
		////////////////////////////////////////////////// Computes E(\Phi \Xi)
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed
		for (it = 0; it < NbInTimes; it++) {
						
			if(ie == NI){
				if(threadIdx.x == 0){
					for(d = 0; d < Dim; d++){
						sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();	

				res1 = -10000.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 = fmaxf(res1, XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total]);
				}
				res1 = fmaxf(res1 - 120.0f, 0.0f) - 2.0f*fmaxf(res1 - 150.0f, 0.0f);

				res = 0.0f;
				for(d = 0;d<Dim; d++){
					for(dd = 0; dd < Dim; dd++){
						res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
						((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
					}
					res += Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter];
				}


				fy = -Rl*res1 - ((mu-Rl)/sigma)*res + (Rb-Rl)*fmaxf(res/sigma - res1, 0.0f);
				res = res1 + dtt*dtt*fy; 
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					}else {
						res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}

					res = 0.0f;
					for(d = 0;d<Dim; d++){
						for(dd = 0; dd < Dim; dd++){
							res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - X_in[n_idx + dd*NbOuter + (ie-1)*NbOuter*Dim] );			
						}
						res += Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter];
					}	

					fy = -Rl*res1 - ((mu-Rl)/sigma)*res + (Rb-Rl)*fmaxf(res/sigma - res1, 0.0f);
					res = res1 + dtt*dtt*fy; 					
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					}else {
						res1 +=  cst[n_idx + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter];
					}	

					res = 0.0f;
					for(d = 0;d<Dim; d++){
						for(dd = 0; dd < Dim; dd++){
							res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - X_in[n_idx + dd*NbOuter + (ie-1)*NbOuter*Dim] );			
						}
						res += Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter];
					}	

					fy = -Rl*res1 - ((mu-Rl)/sigma)*res + (Rb-Rl)*fmaxf(res/sigma - res1, 0.0f);
					res = res1 + dtt*dtt*fy; 	
				}
			}

			if(threadIdx.x == 0){
				for(d = 0; d < Dim; d++){
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
				}
			}			
			__syncthreads();

			for(d = 0; d < Dim; d++){ // Change gamsh to sC or sC2
				gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res-Yt[n_idx + (ie-1)*NbOuter])*
															   (XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total]-
																sR[d + threadIdx.y*Dim])/((float)TotInner);

				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}		
				if(threadIdx.x == 0){
					 atomicAdd(gamY + d + n_idx*Dim + (((2*NI-1-i)*i)/2+(ie-i-2))*NbOuter*Dim, gamsh[threadIdx.y*BlockInnerX]);
				}
			}				
		}
	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);
}

//Coarse and fine approximation kernel for Z (Quadra BSDE)
__global__ void MCZ_Quadra_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					  float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apQ, DP dpQ, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, ii, it, d, dd;
	float g0, g1, res, res1, dw, normX;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	//__shared__ float sC2[BlockInnerX*BlockInnerY];
	int NI = apQ.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (dd = 0; dd < Dim; dd++){					
				res = sR[dd + threadIdx.y*Dim];
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				BoxMuller_d(&g0, &g1);
				Diff_d(&res, dpQ, dtt, g0);
				XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total] = res;
			}


			if(i == NI - 1){
				normX = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normX += powf(XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total], 2.0f);
				}

				res1 = sinf(powf(normX, alph));
			} else {
				if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}					
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter]; 
				} else {
					res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}
				
				
			}	

			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for(d = 0; d < Dim; d++){
				dw = XI_tmp[gb_index_x + d*Total + ((i+1)%2)*Dim*Total]-sR[d + threadIdx.y*Dim];
	
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (1.0f/(dtt*dtt))*res1*dw/((float)TotInner);
				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}
				if (threadIdx.x == 0){
					atomicAdd(Zt + n_idx + d*NbOuter + i*Dim*NbOuter, sC[threadIdx.y*BlockInnerX]);
				}
			}	
		}
	} else{
		/////////////////////////////////// Replaces Matcorr_k i.e. computes E(\Xi ^t\Xi)
		if (threadIdx.x == 0) {
			for (d = 0; d < Dim; d++) {
				sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				sR[d + threadIdx.y*Dim + Dim*BlockInnerY] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
			}
		}
		__syncthreads();
		
		for(it=0; it<NM; it++){
			for (ii = i; ii < ie-1; ii++) { // First ie is equal to NI
				for (d = 0; d < Dim; d++) {
					if (ii == i) {
						res = sR[d + threadIdx.y*Dim];
					}
					else {
						res = XI_tmp[gb_index_x + d*Total + (ii%2)*Dim*Total];
					}
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpQ, dtt, g0);
					XI_tmp[gb_index_x + d*Total + ((ii+1)%2)*Dim*Total] = res;
				}
			}

			for(d = 0; d < Dim; d++){
				for(dd = 0; dd <= d; dd++){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] = (XI_tmp[gb_index_x + d*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[d + threadIdx.y*Dim + Dim*BlockInnerY])
															*(XI_tmp[gb_index_x + dd*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[dd + threadIdx.y*Dim + Dim*BlockInnerY])/(NbMat);
					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}
					if (threadIdx.x == 0){
						atomicAdd(Mat + n_idx*Dim*Dim + dd*Dim + d, sC[threadIdx.y*BlockInnerX]);
						atomicAdd(Mat + n_idx*Dim*Dim + d*Dim + dd, sC[threadIdx.y*BlockInnerX]);
					}
				}
			}

		}

		////////////////////////////////////////////////// Computes E(\Phi \Xi)//////////////////
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed

		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0) {
				for (d = 0; d < Dim; d++) {
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (d = 0; d < Dim; d++) {
				res = sR[d + threadIdx.y*Dim];
				for (ii = i; ii < ie; ii++) { // First ie is equal to NI
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpQ, dtt, g0);
					XI_tmp[gb_index_x + d*Total	+ ((ii + 1) % 2)*Dim*Total] = res;
				}
			}


			if(ie == NI){
				normX = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normX  += powf(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total], 2.0f);
				}
				res1 = sinf(powf(normX, alph)); 					
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					} else {
						res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}
											
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					} else {
						res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}
				}
			}

			for(d = 0; d < Dim; d++){
				dw = XI_tmp[gb_index_x + d*Total + (ie%2)*Dim*Total]-XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total];
				res =  (1.0f/(dtt*dtt))*res1*dw;
				__syncthreads();

				if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();

				for(dd = 0; dd < Dim; dd++){
					gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res - Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter])*
																(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total]-
																				sR[dd + threadIdx.y*Dim])/((float)TotInner);

					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}		
					if(threadIdx.x == 0){
						atomicAdd(gamZ + dd + d*Dim + n_idx*Dim*Dim, gamsh[threadIdx.y*BlockInnerX]);
					}
				}	
			}			
		}

	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

}

//Coarse and fine approximation kernel for Y (Quadra BSDE)
__global__ void MCY_Quadra_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apQ, DP dpQ, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, it, d, dd;
	float res, res1, sto, fy, normXt, normXdt, deltLapPsi,  L;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	float T = dpQ.T;
	int NI = apQ.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if(i == NI - 1){
				normXdt = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normXdt += powf(XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total], 2.0f);
				}
				res1 = sinf(powf(normXdt , alph));

				normXt = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normXt += powf(X_in[n_idx + dd*NbOuter + i*Dim*NbOuter], 2.0f);
				}

				L = 2.0f*alph*powf((T-i*dtt*dtt), (alph-0.5f));
				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					sto = fminf(fmaxf(-L, Zt[n_idx + dd*NbOuter + i*Dim*NbOuter]), L);
					res += powf(sto, 2.0f);
				}

				deltLapPsi = (Dim - 1.0f)*alph*powf(T-i*dtt*dtt + normXt, (alph - 1.0f))*cosf(powf(T-i*dtt*dtt + normXt, alph)) +
				2.0f*alph*(alph - 1.0f)*normXt*powf(T-i*dtt*dtt + normXt, (alph - 2.0f))*cosf(powf(T-i*dtt*dtt + normXt, alph)) -
				2.0f*alph*alph*normXt*powf(T-i*dtt*dtt + normXt, 2.0f*(alph - 1.0f))*sinf(powf(T-i*dtt*dtt + normXt, alph));

				fy = - deltLapPsi;
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner); 
			}else{
			 	if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}	
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter]; 
				} else {
					res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];	
				}	

				
				normXt = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normXt += powf(X_in[n_idx + dd*NbOuter + i*Dim*NbOuter], 2.0f);
				}
				L = 2.0f*alph*powf((T-i*dtt*dtt), (alph-0.5f));
				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					sto = fminf(fmaxf(-L, Zt[n_idx + dd*NbOuter + i*Dim*NbOuter]), L);
					res += powf(sto, 2.0f);
				}

				deltLapPsi = (Dim - 1.0f)*alph*powf(T-i*dtt*dtt + normXt, (alph - 1.0f))*cosf(powf(T-i*dtt*dtt + normXt, alph)) +
				2.0f*alph*(alph - 1.0f)*normXt*powf(T-i*dtt*dtt + normXt, (alph - 2.0f))*cosf(powf(T-i*dtt*dtt + normXt, alph)) -
				2.0f*alph*alph*normXt*powf(T-i*dtt*dtt + normXt, 2.0f*(alph - 1.0f))*sinf(powf(T-i*dtt*dtt + normXt, alph));


				fy =  - deltLapPsi;

				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner);

			}	
			__syncthreads();
			k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
				}
				__syncthreads();
				k /= 2;
			}
			if (threadIdx.x == 0){
				atomicAdd(Yt + n_idx +  i*NbOuter, sC[threadIdx.y*BlockInnerX]);
			}	
		}
	} else{
		////////////////////////////////////////////////// Computes E(\Phi \Xi)
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed
		for (it = 0; it < NbInTimes; it++) {
			if(ie == NI){
				if(threadIdx.x == 0){
					for(d = 0; d < Dim; d++){
						sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();	

				normXdt = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normXdt += powf(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total], 2.0f);
				}
				res1 = sinf(powf(normXdt, alph));


				normXt = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					normXt += powf(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total], 2.0f);
				}

				L = 2.0f*alph*powf((T-(ie-1)*dtt*dtt), (alph-0.5f));
				res = 0.0f;
				for(d = 0;d<Dim; d++){
					sto = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						sto += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
						((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
					}
					sto = fminf(fmaxf(-L, Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + sto), L);
					res += powf(sto, 2.0f);
				}

				deltLapPsi = (Dim - 1.0f)*alph*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 1.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) +
				2.0f*alph*(alph - 1.0f)*normXt*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 2.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) -
				2.0f*alph*alph*normXt*powf(T-(ie-1)*dtt*dtt + normXt, 2.0f*(alph - 1.0f))*sinf(powf(T-(ie-1)*dtt*dtt + normXt, alph));

				fy = - deltLapPsi;
				res = res1 + dtt*dtt*fy; 
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					} else {
						res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];	
					}

					normXt = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						normXt += powf(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total], 2.0f);
					}

					L = 2.0f*alph*powf((T-(ie-1)*dtt*dtt), (alph-0.5f));
					res = 0.0f;
					for(d = 0;d<Dim; d++){
						sto = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							sto += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
						}
						sto = fminf(fmaxf(-L, Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + sto), L);
						res += powf(sto, 2.0f);
					}

					deltLapPsi = (Dim - 1.0f)*alph*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 1.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) +
					2.0f*alph*(alph - 1.0f)*normXt*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 2.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) -
					2.0f*alph*alph*normXt*powf(T-(ie-1)*dtt*dtt + normXt, 2.0f*(alph - 1.0f))*sinf(powf(T-(ie-1)*dtt*dtt + normXt, alph));

					fy =  - deltLapPsi;
					res = res1 + dtt*dtt*fy; 					
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					} else {
						res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];	
					}	

					normXt = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						normXt += powf(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total], 2.0f);
					}

					L = 2.0f*alph*powf((T-(ie-1)*dtt*dtt), (alph-0.5f));
					res = 0.0f;
					for(d = 0;d<Dim; d++){
						sto = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							sto += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
						}
						sto = fminf(fmaxf(-L, Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + sto), L);
						res += powf(sto, 2.0f);
					}

					deltLapPsi = (Dim - 1.0f)*alph*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 1.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) +
					2.0f*alph*(alph - 1.0f)*normXt*powf(T-(ie-1)*dtt*dtt + normXt, (alph - 2.0f))*cosf(powf(T-(ie-1)*dtt*dtt + normXt, alph)) -
					2.0f*alph*alph*normXt*powf(T-(ie-1)*dtt*dtt + normXt, 2.0f*(alph - 1.0f))*sinf(powf(T-(ie-1)*dtt*dtt + normXt, alph));


					fy =  - deltLapPsi;
					res = res1 + dtt*dtt*fy; 					
				}
			}

			if(threadIdx.x == 0){
				for(d = 0; d < Dim; d++){
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
				}
			}			
			__syncthreads();

			for(d = 0; d < Dim; d++){
				gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res-Yt[n_idx + (ie-1)*NbOuter])*
															   (XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total]-
																sR[d + threadIdx.y*Dim])/((float)TotInner);

				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}		
				if(threadIdx.x == 0){
					 atomicAdd(gamY + d + n_idx*Dim + (((2*NI-1-i)*i)/2+(ie-i-2))*NbOuter*Dim, gamsh[threadIdx.y*BlockInnerX]);
				}
			}				
		}
	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);
}

//Coarse and fine approximation kernel for Z (HJB BSDE)
__global__ void MCZ_HJB_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					  float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apH, DP dpH, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, ii, it, d, dd;
	float g0, g1, res, res1, dw;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	int NI = apH.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (dd = 0; dd < Dim; dd++){					
				res = sR[dd + threadIdx.y*Dim];
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				BoxMuller_d(&g0, &g1);
				Diff_d(&res, dpH, dtt, g0);
				XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total] = res;
			}

			if(i == NI - 1){
				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total], 2.0f);
				}

				res1 = logf(0.5f*(1.0f + res1));
			} else {
				if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}		
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter];
				} else {
					res1 += cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}
			}	


			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for(d = 0; d < Dim; d++){
				dw = (XI_tmp[gb_index_x + d*Total + ((i+1)%2)*Dim*Total]-sR[d + threadIdx.y*Dim])/sqrtf(2.0f);
	
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (1.0f/(dtt*dtt))*res1*dw/((float)TotInner);
				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}
				if (threadIdx.x == 0){
					atomicAdd(Zt + n_idx + d*NbOuter + i*Dim*NbOuter, sC[threadIdx.y*BlockInnerX]);
				}
			}	
		}
	} else{
		/////////////////////////////////// Replaces Matcorr_k i.e. computes E(\Xi ^t\Xi)
		if (threadIdx.x == 0) {
			for (d = 0; d < Dim; d++) {
				sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				sR[d + threadIdx.y*Dim + Dim*BlockInnerY] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
			}
		}
		__syncthreads();
		
		for(it=0; it<NM; it++){
			for (ii = i; ii < ie-1; ii++) { // First ie is equal to NI
				for (d = 0; d < Dim; d++) {
					if (ii == i) {
						res = sR[d + threadIdx.y*Dim];
					}
					else {
						res = XI_tmp[gb_index_x + d*Total + (ii%2)*Dim*Total];
					}
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpH, dtt, g0);
					XI_tmp[gb_index_x + d*Total + ((ii+1)%2)*Dim*Total] = res;
				}
			}

			for(d = 0; d < Dim; d++){
				for(dd = 0; dd <= d; dd++){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] = (XI_tmp[gb_index_x + d*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[d + threadIdx.y*Dim + Dim*BlockInnerY])
															*(XI_tmp[gb_index_x + dd*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[dd + threadIdx.y*Dim + Dim*BlockInnerY])/(NbMat);
					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}
					if (threadIdx.x == 0){
						atomicAdd(Mat + n_idx*Dim*Dim + dd*Dim + d, sC[threadIdx.y*BlockInnerX]);
						atomicAdd(Mat + n_idx*Dim*Dim + d*Dim + dd, sC[threadIdx.y*BlockInnerX]);
					}
				}
			}

		}

		////////////////////////////////////////////////// Computes E(\Phi \Xi)//////////////////
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed

		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0) {
				for (d = 0; d < Dim; d++) {
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (d = 0; d < Dim; d++) {
				res = sR[d + threadIdx.y*Dim];
				for (ii = i; ii < ie; ii++) { // First ie is equal to NI
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpH, dtt, g0);
					XI_tmp[gb_index_x + d*Total	+ ((ii + 1) % 2)*Dim*Total] = res;
				}
			}


			if(ie == NI){
				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total], 2.0f);
				}

				res1 = logf(0.5f*(1.0f + res1)); 					
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter];
					} else {
						res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}										
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter];
					} else {
						res1 +=  cst[n_idx + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter];
					}										
				}
			}
			__syncthreads();

			if(threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + (ie-1)*NbOuter*Dim];
				}
			}			
			__syncthreads();
				
			for(d = 0; d < Dim; d++){
				dw = (XI_tmp[gb_index_x + d*Total + (ie%2)*Dim*Total]-
				XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total])/sqrtf(2.0f);

				res = (1.0f/(dtt*dtt))*res1*dw;
				for(dd = 0; dd < Dim; dd++){ // Change gamsh to sC or sC2
					gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res - Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter])*
																(XI_tmp[gb_index_x + dd*Total + ((ie-1)%2)*Dim*Total]-
																				sR[dd + threadIdx.y*Dim])/((float)TotInner);

					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}		
					if(threadIdx.x == 0){
						atomicAdd(gamZ + dd + d*Dim + n_idx*Dim*Dim, gamsh[threadIdx.y*BlockInnerX]);
					}
				}	
			}			
		}

	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

}

//Coarse and fine approximation kernel for Y (HJB BSDE)
__global__ void MCY_HJB_k1(float *XI_tmp, float *X_in, float *Yt, float *Zt, int i, int ie, float dtt,
					float *gamY, float *gamZ, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apH, DP dpH, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, ii, it, d, dd;
	float g0, g1, res, res1, fy;   
	__shared__ float sR[Dim*BlockInnerY];
	__shared__ float sR2[Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	int NI = apH.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (dd = 0; dd < Dim; dd++){					
				res = sR[dd + threadIdx.y*Dim];
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				BoxMuller_d(&g0, &g1);
				Diff_d(&res, dpH, dtt, g0);
				XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total] = res;
			}

			if(i == NI - 1){
				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total], 2.0f);
				}
				res1 = logf(0.5f*(1.0f + res1));

				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res += powf(Zt[n_idx + dd*NbOuter + i*Dim*NbOuter], 2.0f);
				}

				fy = -res;
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner); 
			}else{
				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}					
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter];
				} else {
					res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}	

				res = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res += powf(Zt[n_idx + dd*NbOuter + i*Dim*NbOuter], 2.0f);
				}

				fy = -res;

				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner);
			}	
			__syncthreads();
			k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
				}
				__syncthreads();
				k /= 2;
			}
			if (threadIdx.x == 0){
				atomicAdd(Yt + n_idx +  i*NbOuter, sC[threadIdx.y*BlockInnerX]);
			}	
		}
	} else{
		////////////////////////////////////////////////// Computes E(\Phi \Xi)
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0) {
				for (d = 0; d < Dim; d++) {
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (d = 0; d < Dim; d++) {
				res = sR[d + threadIdx.y*Dim];
				for (ii = i; ii < ie; ii++) { // First ie is equal to NI
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpH, dtt, g0);
					XI_tmp[gb_index_x + d*Total	+ ((ii + 1) % 2)*Dim*Total] = res;
				}
			}

			if(ie == NI){
				if(threadIdx.x == 0){
					for(d = 0; d < Dim; d++){
						sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();	

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total], 2.0f);
				}
				res1 = logf(0.5f*(1.0f + res1));

				for(d = 0;d<Dim; d++){
					res = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
						((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
					}
					sR2[d + threadIdx.y*Dim] = Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + res;
				}

				res = 0.0f;
				for(int d = 0; d < Dim; d++){
					res += powf(sR2[d + threadIdx.y*Dim], 2.0f); 
				}
				fy = -res;
				res = res1 + dtt*dtt*fy; 
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter];
					} else {
						res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}

					for(d = 0; d<Dim; d++){
						res = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
						}
						sR2[d + threadIdx.y*Dim] = Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + res;
					}

					res = 0.0f;
					for(int d = 0; d < Dim; d++){
						res += powf(sR2[d + threadIdx.y*Dim], 2.0f); 
					}

					fy = -res;
					res = res1 + dtt*dtt*fy; 					
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();


					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter];
					} else {
						res1 += cst[n_idx + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter];
					}

					for(d = 0;d<Dim; d++){
						res = 0.0f;
						for(dd = 0; dd < Dim; dd++){
							res += gamZ[dd + d*Dim + n_idx*Dim*Dim]*(XI_tmp[gb_index_x + dd*Total + 
							((ie-1)%2)*Dim*Total] - sR[dd + threadIdx.y*Dim] );			
						}
						sR2[d + threadIdx.y*Dim] = Zt[n_idx + d*NbOuter + (ie-1)*Dim*NbOuter] + res;
					}

					res = 0.0f;
					for(int d = 0; d < Dim; d++){
						res += powf(sR2[d + threadIdx.y*Dim], 2.0f); 
					}

					fy = -res;
					res = res1 + dtt*dtt*fy; 	
				}
			}

			if(threadIdx.x == 0){
				for(d = 0; d < Dim; d++){
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
				}
			}			
			__syncthreads();

			for(d = 0; d < Dim; d++){ // Change gamsh to sC or sC2
				gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res-Yt[n_idx + (ie-1)*NbOuter])*
															   (XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total]-
																sR[d + threadIdx.y*Dim])/((float)TotInner);

				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}		
				if(threadIdx.x == 0){
					 atomicAdd(gamY + d + n_idx*Dim + (((2*NI-1-i)*i)/2+(ie-i-2))*NbOuter*Dim, gamsh[threadIdx.y*BlockInnerX]);
				}
			}				
		}
	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

}

//Coarse and fine approximation kernel for Y (AllenCahn BSDE)
__global__ void MCY_AllenCahn_k1(float *XI_tmp, float *X_in, float *Yt, int i, int ie, float dtt,
					float *gamY, float *Mat, float *cst, TabSeedCMRG_t *pt_cmrg, AP apA, DP dpA, int ni, int flag){

    int xy_idx = threadIdx.x + blockIdx.y*blockDim.x;
	int n_idx = threadIdx.y + blockIdx.x*blockDim.y;
	int gb_index_x = xy_idx + n_idx*gridDim.y*blockDim.x;	

	int a0, a1, a2, a3, a4, a5, k, ii, it, d, dd;
	float g0, g1, res, res1, fy;   
	__shared__ float sR[2*Dim*BlockInnerY];
	__shared__ float sC[BlockInnerX*BlockInnerY];
	int NI = apA.NI;

    CMRG_get_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);

	if(i == NI-1 || ie == i+1){
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0){
				for(dd = 0; dd < Dim; dd++){
					sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (dd = 0; dd < Dim; dd++){					
				res = sR[dd + threadIdx.y*Dim];
				CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
				BoxMuller_d(&g0, &g1);
				Diff_d(&res, dpA, dtt, g0);
				XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total] = res;
			}			
			if(i == NI - 1){
				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + ((i+1)%2)*Dim*Total], 2.0f);
				}
				res1 = 0.5f/(1.0f + 0.2f*res1);


				fy = res1 - powf(res1, 3.0f);
				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner); 
			}else{
			 	if(threadIdx.x == 0){
					for(dd = 0; dd < Dim; dd++){
						sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
					}
				}
				__syncthreads();

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += gamY[(((2*NI-1-i)*i)/2)*NbOuter*Dim + Dim*n_idx + dd]*
					(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
				}					
				if(flag == 0){
					res1 += Yt[n_idx + ie*NbOuter]; 
				}else{ 
					res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
				}	

				fy =  res1 - powf(res1, 3.0f);

				sC[threadIdx.x + threadIdx.y*BlockInnerX] = (res1 + dtt*dtt*fy)/((float)TotInner);
			}	
			__syncthreads();
			k = blockDim.x / 2;
			while (k != 0) {
				if (threadIdx.x < k){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
				}
				__syncthreads();
				k /= 2;
			}
			if (threadIdx.x == 0){
				atomicAdd(Yt + n_idx +  i*NbOuter, sC[threadIdx.y*BlockInnerX]);
			}	
		}
	} else{
		/////////////////////////////////// Replaces Matcorr_k i.e. computes E(\Xi ^t\Xi)
		if (threadIdx.x == 0) {
			for (d = 0; d < Dim; d++) {
				sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				sR[d + threadIdx.y*Dim + Dim*BlockInnerY] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
			}
		}
		__syncthreads();
		
		for(it=0; it<NM; it++){
			for (ii = i; ii < ie-1; ii++) { // First ie is equal to NI
				for (d = 0; d < Dim; d++) {
					if (ii == i) {
						res = sR[d + threadIdx.y*Dim];
					}
					else {
						res = XI_tmp[gb_index_x + d*Total + (ii%2)*Dim*Total];
					}
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpA, dtt, g0);
					XI_tmp[gb_index_x + d*Total + ((ii+1)%2)*Dim*Total] = res;
				}
			}

			for(d = 0; d < Dim; d++){
				for(dd = 0; dd <= d; dd++){
					sC[threadIdx.x + threadIdx.y*BlockInnerX] = (XI_tmp[gb_index_x + d*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[d + threadIdx.y*Dim + Dim*BlockInnerY])
															*(XI_tmp[gb_index_x + dd*Total + ((ie - 1) % 2)*Dim*Total]-
															sR[dd + threadIdx.y*Dim + Dim*BlockInnerY])/(NbMat);
					__syncthreads();
					k = blockDim.x / 2;
					while (k != 0) {
						if (threadIdx.x < k){
							sC[threadIdx.x + threadIdx.y*BlockInnerX] += sC[threadIdx.x + threadIdx.y*BlockInnerX + k];
						}
						__syncthreads();
						k /= 2;
					}
					if (threadIdx.x == 0){
						atomicAdd(Mat + n_idx*Dim*Dim + dd*Dim + d, sC[threadIdx.y*BlockInnerX]);
						atomicAdd(Mat + n_idx*Dim*Dim + d*Dim + dd, sC[threadIdx.y*BlockInnerX]);
					}
				}
			}

		}

		////////////////////////////////////////////////// Computes E(\Phi \Xi)
		__shared__ float gamsh[BlockInnerX*BlockInnerY]; // gamsh not needed
		for (it = 0; it < NbInTimes; it++) {
			if (threadIdx.x == 0) {
				for (d = 0; d < Dim; d++) {
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + i*NbOuter*Dim];
				}
			}
			__syncthreads();

			for (d = 0; d < Dim; d++) {
				res = sR[d + threadIdx.y*Dim];
				for (ii = i; ii < ie; ii++) { // First ie is equal to NI
					CMRG_d(&a0, &a1, &a2, &a3, &a4, &a5, &g0, &g1, 2);
					BoxMuller_d(&g0, &g1);
					Diff_d(&res, dpA, dtt, g0);
					XI_tmp[gb_index_x + d*Total	+ ((ii + 1) % 2)*Dim*Total] = res;
				}
			}

			if(ie == NI){
				if(threadIdx.x == 0){
					for(d = 0; d < Dim; d++){
						sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
					}
				}			
				__syncthreads();	

				res1 = 0.0f;
				for(dd = 0; dd < Dim; dd++){
					res1 += powf(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total], 2.0f);
				}
				res1 = 0.5f/(1.0f + 0.2f*res1);


				fy = res1 - powf(res1, 3.0f);
				res = res1 + dtt*dtt*fy; 
			} else {
				if((ni==NI) || (ie<ni)){
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-i)*i)/2 + ie-i-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					}else{ 
						res1 +=  cst[n_idx + (((2*NI-1-i)*i)/2+(ie-i-1))*NbOuter];
					}

					fy = res1 - powf(res1, 3.0f);
					res = res1 + dtt*dtt*fy; 					
				} else {
					if(threadIdx.x == 0){
						for(dd = 0; dd < Dim; dd++){
							sR[dd + threadIdx.y*Dim] = X_in[n_idx + dd*NbOuter + ie*NbOuter*Dim];
						}
					}
					__syncthreads();

					res1 = 0.0f;
					for(dd = 0; dd < Dim; dd++){
						res1 += gamY[dd + n_idx*Dim + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter*Dim]*
								(XI_tmp[gb_index_x + dd*Total + (ie%2)*Dim*Total] - sR[dd + threadIdx.y*Dim]);
					}	
					if(flag == 0){
						res1 += Yt[n_idx + ie*NbOuter]; 
					}else{ 
						res1 +=  cst[n_idx + (((2*NI-1-(i+1))*(i+1))/2 + ie-(i+1)-1)*NbOuter];
					}


					fy = res1 - powf(res1, 3.0f);
					res = res1 + dtt*dtt*fy; 	
				}
			}

			if(threadIdx.x == 0){
				for(d = 0; d < Dim; d++){
					sR[d + threadIdx.y*Dim] = X_in[n_idx + d*NbOuter + (ie-1)*NbOuter*Dim];
				}
			}			
			__syncthreads();

			for(d = 0; d < Dim; d++){ 
				gamsh[threadIdx.x + threadIdx.y*BlockInnerX] = (res-Yt[n_idx + (ie-1)*NbOuter])*
															   (XI_tmp[gb_index_x + d*Total + ((ie-1)%2)*Dim*Total]-
																sR[d + threadIdx.y*Dim])/((float)TotInner);

				__syncthreads();
				k = blockDim.x / 2;
				while (k != 0) {
					if (threadIdx.x < k){
						gamsh[threadIdx.x + threadIdx.y*BlockInnerX] += gamsh[threadIdx.x + threadIdx.y*BlockInnerX + k];
					}
					__syncthreads();
					k /= 2;
				}		
				if(threadIdx.x == 0){
					 atomicAdd(gamY + d + n_idx*Dim + (((2*NI-1-i)*i)/2+(ie-i-2))*NbOuter*Dim, gamsh[threadIdx.y*BlockInnerX]);
				}
			}				
		}
	}

	CMRG_set_d(&a0, &a1, &a2, &a3, &a4, &a5, pt_cmrg[0][xy_idx][n_idx]);
}

// Optimized for matrices bigger than 64x64: x64 number of threads
//LDLt_multi_k (Matcorr, GamZGPU , ii-1, Dim, Dim);
__global__ void LDLt_multi_k(float *a, float *y, int t, int ii, int n, int nbt)
{
	// Identifies the thread working within a group
	int tidx = threadIdx.x%n;
	// Shared memory
	extern __shared__ float sA[];
	// Local integers
	int i, n2, j;

	//j = (((2*NI-1-t)*t)/2 +(ii-t -1))*gridDim.x*n;
	j = 0;
	n2 = (n*n + n) / 2;

	for (i = n; i>0; i--){
		if (tidx<i){
			sA[n2 - i*(i + 1) / 2 + tidx] = a[blockIdx.x*n*n + (n - i)*(n + 1) + tidx];
		}
	}
	__syncthreads();


	for (i = n; i>0; i--){
		if (tidx == 0){
			for (int k = n; k>i; k--){
				sA[n2 - i*(i + 1) / 2] -= sA[n2 - k*(k + 1) / 2] *
					sA[n2 - k*(k + 1) / 2 + k - i] *
					sA[n2 - k*(k + 1) / 2 + k - i];
			}
		}
		__syncthreads();
		if (tidx<i - 1){
			sA[n2 - i*(i + 1) / 2 + tidx + 1] /= sA[n2 - i*(i + 1) / 2];
			for (int k = n; k>i; k--){
				sA[n2 - i*(i + 1) / 2 + tidx + 1] -= sA[n2 - k*(k + 1) / 2] *
					sA[n2 - k*(k + 1) / 2 + k - i] *
					sA[n2 - k*(k + 1) / 2 + tidx + 1 + k - i] /
					sA[n2 - i*(i + 1) / 2];
			}
		}
		__syncthreads();
	}


	for (i = 0; i < n - 1; i++){
		if (tidx > i){		
			y[j + blockIdx.x*n*nbt + blockIdx.y*n + tidx] -= sA[n2 - (n - i)*(n - i + 1) / 2 + tidx - i] *
															y[j + blockIdx.x*n*nbt + blockIdx.y*n + i];
		}
		__syncthreads();
	}

	y[j + blockIdx.x*n*nbt + blockIdx.y*n + tidx] /= sA[n2 - (n - tidx)*(n - tidx + 1) / 2];
	__syncthreads();

	for (i = n - 1; i > 0; i--){
		if (tidx < i){
			y[j + blockIdx.x*n*nbt + blockIdx.y*n + tidx] -= sA[n2 - (n - tidx)*(n - tidx + 1) / 2 + i - tidx] *
				y[j + blockIdx.x*n*nbt + blockIdx.y*n + i];
		}
		__syncthreads();
	}
}

// Optimized for matrices bigger than 64x64: x64 number of threads
//LDLt_max_global_k<<<NbOuter, Dim>>>(Matcorr, GamGPU, i, ii, Dim);
__global__ void LDLt_max_global_k(float *a, float *y, int t, int ii, int n, AP ap)
{
    // Identifies the thread working within a group
    int tidx = threadIdx.x%n;
    // Identifies the data concerned by the computations
    int Qt = (threadIdx.x-tidx)/n;
    // The global memory access index
    int gb_index_x = Qt + blockIdx.x*(blockDim.x/n);
    // Local integers
    int i, k, j;
	int NI = ap.NI;

	j = (((2*NI-1-t)*t)/2 +(ii-t -1))*gridDim.x*n;

    // Perform the LDLt factorization
    for(i=n; i>0; i--){
        if(tidx==0){
            for(k=n; k>i; k--){
				a[gb_index_x*n*n+(n-i)*(n+1)] -= 
				a[gb_index_x*n*n+(n-k)*(n+1)]*
				a[gb_index_x*n*n+(n-k)*(n+1)+k-i]*
				a[gb_index_x*n*n+(n-k)*(n+1)+k-i]; 
            }
        }
        __syncthreads();
        if(tidx<i-1){
			a[gb_index_x*n*n+(n-i)*(n+1)+tidx+1] /=
			a[gb_index_x*n*n+(n-i)*(n+1)];
            for(k=n; k>i; k--){
				a[gb_index_x*n*n+(n-i)*(n+1)+tidx+1] -=
				a[gb_index_x*n*n+(n-k)*(n+1)]*
				a[gb_index_x*n*n+(n-k)*(n+1)+k-i]*
				a[gb_index_x*n*n+(n-k)*(n+1)+tidx+1+k-i]/
				a[gb_index_x*n*n+(n-i)*(n+1)];
            }
        }
        __syncthreads();
    }

    // Resolve the system using LDLt factorization
    for(i=0; i<n-1; i++){
        if(tidx>i){		
			//y[(((2*NI-1-(t/S))*(t/S))/2 +((ii/S)-(t/S) -1))*gridDim.x*n+gb_index_x*n+tidx] -=
			y[j + gb_index_x*n + tidx] -= a[gb_index_x*n*n+(i)*(n+1)+tidx-i]*y[j + gb_index_x*n + i];
			//y[(((2*NI-1-(t/S))*(t/S))/2 +((ii/S)-(t/S) -1))*gridDim.x*n+gb_index_x*n+i];			  
        }
        __syncthreads();
    }	

	y[j + gb_index_x*n + tidx] /= a[gb_index_x*n*n+(tidx)*(n+1)];
    __syncthreads();
    for(i=n-1; i>0; i--){
        if(tidx<i){
			y[j + gb_index_x*n + tidx] -= a[gb_index_x*n*n+(tidx)*(n+1)+i-tidx]*y[j + gb_index_x*n + i];
        }
        __syncthreads();
    }
}


//Monte Carlo for Y value at time 0
__global__ void Expect0_k(float *Yt, float *val, float *var_, int flag){
	int gb_index_x = threadIdx.x + blockIdx.x*blockDim.x; //Index
	__shared__ float sC[BlockOuter];
	__shared__ float sCS[BlockOuter];
	int k;
	sC[threadIdx.x] = Yt[gb_index_x +flag*blockDim.x*gridDim.x]/((float)NbOuter);
	sCS[threadIdx.x] = sC[threadIdx.x]*sC[threadIdx.x]*((float)NbOuter);
	__syncthreads();
	k = blockDim.x / 2;
	while (k != 0) {
		if (threadIdx.x < k){
			sC[threadIdx.x] += sC[threadIdx.x + k];
			sCS[threadIdx.x] += sCS[threadIdx.x + k];
		}
		__syncthreads();
		k /= 2;
	}
	if(threadIdx.x == 0){
		atomicAdd(val, sC[0]);
		atomicAdd(var_, sCS[0]);
		
		//printf("\n %f, ", val[0]);
	}

}
/***************************************************************
*         *  *****      *      *****  *****  *****  *****  *****
*         *  *   *     * *     *   *  *   *  *      *   *  *
*   *   *   ****     *   *    *****  *****  *****  ****   *****
* * * *    *   *   *******   *      *      *      *   *      *
*   *     *   *  *       *  *      *      *****  *   *  *****
***************************************************************/
void CondMCLearn_BSDE_Bergam(AP APB, DP DPB, float dtt, int flag){

	///////Block and threads management////////
	dim3 Dg(GridOuter, 1, 1);
	dim3 Db(BlockOuter, 1, 1);

	dim3 DgI(GridInnerX, GridInnerY, 1);
	dim3 DbI(BlockInner, 1, 1);

	dim3 DgII(NbOuter, 1, 1);
	dim3 DbII(BlockInner, 1, 1);

	dim3 DgB(GridInnerX, GridInnerY, 1);
	dim3 DbB(BlockInnerX, BlockInnerY, 1);		

	dim3 Dg2(NbOuter, Dim, 1);
	///////////////////////////////////////////
	int NI = APB.NI;

	/////////////Array initialization///////////////////////
	testCUDA(cudaHostGetDevicePointer(&GamGPUZ, GammaZ, 0));
	testCUDA(cudaHostGetDevicePointer(&GamGPUY, GammaY, 0));
	testCUDA(cudaMemset(GamGPUY, 0.0f, ((NI*(NI-1))/2)*NbOuter*Dim*sizeof(float)));
	testCUDA(cudaMemset(Y, 0.0f, (NI+1)*NbOuter*sizeof(float)));
	testCUDA(cudaMemset(Z, 0.0f, NI*Dim*NbOuter*sizeof(float)));
    ////////////////////////////////////////////////////////


	//Launch outer trajectories 
	GeneratePathsOuter_k<<<Dg, Db>>>(X, dtt, pt_CMRG, APB, DPB, flag);

	int ni = NI;
    for(int i = NI-1; i>=0; i--){
		printf("\n TIME step %i: ", i);	
		for(int ii = ni; ii>i; ii--){
			testCUDA(cudaMemset(Matcorr, 0.0f, NbOuter*Dim*Dim*sizeof(float)));
			testCUDA(cudaMemset(GamGPUZ, 0.0f, NbOuter*Dim*Dim*sizeof(float)));

			//Main kernel for coarse and fine approximation of Z
			MCZ_Bergam_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APB, DPB, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix
				LDLt_multi_k<<<Dg2, Dim, ((Dim*Dim + Dim)/2)*sizeof(float)>>> (Matcorr, GamGPUZ , i, ii-1, Dim, Dim);
			}	

			//Main kernel for coarse and fine approximation of Y
			MCY_Bergam_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APB, DPB, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix
				LDLt_max_global_k<<<NbOuter, Dim>>>(Matcorr, GamGPUY, i, ii-1, Dim, APB);
			}	

		}
	}

	testCUDA(cudaMemset(val1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(val2, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var2, 0.0f, sizeof(float)));

	Expect0_k<<<Dg, Db>>>(Y, val1, var1 , 0);//Learned

	Expect0_k<<<Dg, Db>>>(Y, val2, var2 , 1);//Simulated
}

void CondMCLearn_BSDE_Quadra(AP APQ, DP DPQ, float dtt, int flag){

	///////Block and threads management////////
	dim3 Dg(GridOuter, 1, 1);
	dim3 Db(BlockOuter, 1, 1);

	dim3 DgI(GridInnerX, GridInnerY, 1);
	dim3 DbI(BlockInner, 1, 1);

	dim3 DgII(NbOuter, 1, 1);
	dim3 DbII(BlockInner, 1, 1);

	dim3 DgB(GridInnerX, GridInnerY, 1);
	dim3 DbB(BlockInnerX, BlockInnerY, 1);		

	dim3 Dg2(NbOuter, Dim, 1);
	///////////////////////////////////////////
	int NI = APQ.NI;

	/////////////Array initialization///////////////////////
	testCUDA(cudaHostGetDevicePointer(&GamGPUZ, GammaZ, 0));
	testCUDA(cudaHostGetDevicePointer(&GamGPUY, GammaY, 0));
	testCUDA(cudaMemset(GamGPUY, 0.0f, ((NI*(NI-1))/2)*NbOuter*Dim*sizeof(float)));
	testCUDA(cudaMemset(Y, 0.0f, (NI+1)*NbOuter*sizeof(float)));
	testCUDA(cudaMemset(Z, 0.0f, NI*Dim*NbOuter*sizeof(float)));
	/////////////////////////////////////////////////////////
		
	//Launch outer trajectories 
	GeneratePathsOuter_k<<<Dg, Db>>>(X, dtt, pt_CMRG, APQ, DPQ, flag);

	int ni = NI;
    for(int i = NI-1; i>=0; i--){
		printf("\n TIME step %i: ", i);	
		for(int ii = ni; ii>i; ii--){
			testCUDA(cudaMemset(Matcorr, 0.0f, NbOuter*Dim*Dim*sizeof(float)));
			testCUDA(cudaMemset(GamGPUZ, 0.0f, NbOuter*Dim*Dim*sizeof(float)));

			//Main kernel for coarse and fine approximation of Z
			MCZ_Quadra_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APQ, DPQ, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix (multi matrix)
				LDLt_multi_k<<<Dg2, Dim, ((Dim*Dim + Dim)/2)*sizeof(float)>>> (Matcorr, GamGPUZ , i, ii-1, Dim, Dim);
			}	

			//Main kernel for coarse and fine approximation of Y
			MCY_Quadra_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APQ, DPQ, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix 
				LDLt_max_global_k<<<NbOuter, Dim>>>(Matcorr, GamGPUY, i, ii-1, Dim, APQ);
			}	

		}


		////////////////////// Bias control (by cutting the regression error propagation)
		//Not really need in this example but allows to reduce time computation
		if(i< NI-3){
				ni = i + 3;
		}

	}	

	testCUDA(cudaMemset(val1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(val2, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var2, 0.0f, sizeof(float)));

	Expect0_k<<<Dg, Db>>>(Y, val1, var1 , 0);//Méthode 1

	Expect0_k<<<Dg, Db>>>(Y, val2, var2 , 1);//Méthode 2

}

void CondMCLearn_BSDE_AllenCahn(AP APA, DP DPA, float dtt, int flag){

	///////Block and threads management////////
	dim3 Dg(GridOuter, 1, 1);
	dim3 Db(BlockOuter, 1, 1);

	dim3 DgI(GridInnerX, GridInnerY, 1);
	dim3 DbI(BlockInner, 1, 1);

	dim3 DgII(NbOuter, 1, 1);
	dim3 DbII(BlockInner, 1, 1);

	dim3 DgB(GridInnerX, GridInnerY, 1);
	dim3 DbB(BlockInnerX, BlockInnerY, 1);		

	dim3 Dg2(NbOuter, Dim, 1);
	///////////////////////////////////////////
	int NI = APA.NI;

	/////////////Array initialization///////////////////////
	testCUDA(cudaHostGetDevicePointer(&GamGPUY, GammaY, 0));
	testCUDA(cudaMemset(GamGPUY, 0.0f, ((NI*(NI-1))/2)*NbOuter*Dim*sizeof(float)));
	testCUDA(cudaMemset(Y, 0.0f, (NI+1)*NbOuter*sizeof(float)));
    ////////////////////////////////////////////////////////


	//Launch outer trajectories 
	GeneratePathsOuter_k<<<Dg, Db>>>(X, dtt, pt_CMRG, APA, DPA, flag);

	int ni = NI;
    for(int i = NI-1; i>=0; i--){
		printf("\n TIME step %i: ", i);	
		for(int ii = ni; ii>i; ii--){
			testCUDA(cudaMemset(Matcorr, 0.0f, NbOuter*Dim*Dim*sizeof(float)));

			//Main kernel for coarse and fine approximation of Y
			MCY_AllenCahn_k1<<<DgB, DbB>>>(XI, X, Y, i, ii, dtt, GamGPUY, Matcorr, CstGPU, pt_CMRG, APA, DPA, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix 
				LDLt_max_global_k<<<NbOuter, Dim>>>(Matcorr, GamGPUY, i, ii-1, Dim, APA);
			}	

		}
	}	

	testCUDA(cudaMemset(val1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(val2, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var2, 0.0f, sizeof(float)));

	Expect0_k<<<Dg, Db>>>(Y, val1, var1 , 0);//Méthode 1

	Expect0_k<<<Dg, Db>>>(Y, val2, var2 , 1);//Méthode 2


}

void CondMCLearn_BSDE_HJB(AP APH, DP DPH, float dtt, int flag){
	///////Block and threads management////////
	dim3 Dg(GridOuter, 1, 1);
	dim3 Db(BlockOuter, 1, 1);

	dim3 DgI(GridInnerX, GridInnerY, 1);
	dim3 DbI(BlockInner, 1, 1);

	dim3 DgII(NbOuter, 1, 1);
	dim3 DbII(BlockInner, 1, 1);

	dim3 DgB(GridInnerX, GridInnerY, 1);
	dim3 DbB(BlockInnerX, BlockInnerY, 1);		

	dim3 Dg2(NbOuter, Dim, 1);
	///////////////////////////////////////////
	int NI = APH.NI;

	/////////////Array initialization///////////////////////
	testCUDA(cudaHostGetDevicePointer(&GamGPUZ, GammaZ, 0));
	testCUDA(cudaHostGetDevicePointer(&GamGPUY, GammaY, 0));
	testCUDA(cudaMemset(GamGPUY, 0.0f, ((NI*(NI-1))/2)*NbOuter*Dim*sizeof(float)));
	testCUDA(cudaMemset(Y, 0.0f, (NI+1)*NbOuter*sizeof(float)));
	testCUDA(cudaMemset(Z, 0.0f, NI*Dim*NbOuter*sizeof(float)));
    ////////////////////////////////////////////////////////


	//Launch outer trajectories 
	GeneratePathsOuter_k<<<Dg, Db>>>(X, dtt, pt_CMRG, APH, DPH, flag);

	int ni = NI;
    for(int i = NI-1; i>=0; i--){
		printf("\n TIME step %i: ", i);	
		for(int ii = ni; ii>i; ii--){
			testCUDA(cudaMemset(Matcorr, 0.0f, NbOuter*Dim*Dim*sizeof(float)));
			testCUDA(cudaMemset(GamGPUZ, 0.0f, NbOuter*Dim*Dim*sizeof(float)));

			//Main kernel for coarse and fine approximation of Z
			MCZ_HJB_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APH, DPH, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix
				LDLt_multi_k<<<Dg2, Dim, ((Dim*Dim + Dim)/2)*sizeof(float)>>> (Matcorr, GamGPUZ , i, ii-1, Dim, Dim);
			}	

			//Main kernel for coarse and fine approximation of Y
			MCY_HJB_k1<<<DgB, DbB>>>(XI, X, Y, Z, i, ii, dtt, GamGPUY, GamGPUZ, Matcorr, CstGPU, pt_CMRG, APH, DPH, ni, 0);
			if((i != NI-1) && (ii != i+1)){
				//LDLt decomposition of the correlated matrix
				LDLt_max_global_k<<<NbOuter, Dim>>>(Matcorr, GamGPUY, i, ii-1, Dim, APH);
			}	
		}

		if(i < NI-1){
			ni = i+1;
		}
	}

	testCUDA(cudaMemset(val1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var1, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(val2, 0.0f, sizeof(float)));
	testCUDA(cudaMemset(var2, 0.0f, sizeof(float)));

	Expect0_k<<<Dg, Db>>>(Y, val1, var1 , 0);//Learned

	Expect0_k<<<Dg, Db>>>(Y, val2, var2 , 1);//Simulated
}
















