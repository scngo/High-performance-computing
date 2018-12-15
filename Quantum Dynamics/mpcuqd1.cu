/*******************************************************************************
Quantum dynamics (QD) simulation of an electron in one dimension.

USAGE

%cc -o qd1 qd1.c -lm
%qd1 < qd1.in (see qd1.h for the input-file format)

mpicc mpqd1.c -o mpcuqd1 -lm
nvcc -I/usr/usc/openmpi/default/include -L/usr/usc/openmpi/default/lib -lmpi mpcuqd1.cu -o mpcuqd1
*******************************************************************************/
#include <stdio.h>
#include <math.h>
#include "mpcuqd1.h"

int main(int argc, char **argv) {
  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);
  MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
  int step; /* Simulation loop iteration index */

  init_param();   /* Read input parameters */
  init_prop();    /* Initialize the kinetic & potential propagators */
  init_wavefn();  /* Initialize the electron wave function */
  init_gpu(myid); /* Initialize the GPU arrays */

  for (step=1; step<=NSTEP; step++) {
    single_step(); /* Time propagation for one step, DT */
    if (step%NECAL==0) {
      calc_energy();
      if (myid==0) printf("%le %le %le %le\n",DT*step,ekin_global,epot_global,etot_global);
    }
  }

  MPI_Finalize();
  return 0;
}

/*------------------------------------------------------------------------------
	Initializes parameters by reading them from standard input.
------------------------------------------------------------------------------*/
void init_param() {
	/* Read control parameters */
	FILE *input;
	input = fopen("qd1.in","r");
	fscanf(input,"%le",&LX);
	fscanf(input,"%le",&DT);
	fscanf(input,"%d",&NSTEP);
	fscanf(input,"%d",&NECAL);
	fscanf(input,"%le%le%le",&X0,&S0,&E0);
	fscanf(input,"%le%le",&BH,&BW);
	fscanf(input,"%le",&EH);
	fclose(input);

	/* Calculate the mesh size */
	dx = LX/NX;
}

/*------------------------------------------------------------------------------
	Initializes the kinetic & potential propagators.
------------------------------------------------------------------------------*/
void init_prop() {
	int stp,s,i,up,lw;
	double a,exp_p[2],ep[2],em[2];
	double x;

	/* Set up kinetic propagators */
	a = 0.5/(dx*dx);

	for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
		exp_p[0] = cos(-(stp+1)*DT*a);
		exp_p[1] = sin(-(stp+1)*DT*a);
		ep[0] = 0.5*(1.0+exp_p[0]);
		ep[1] = 0.5*exp_p[1];
		em[0] = 0.5*(1.0-exp_p[0]);
		em[1] = -0.5*exp_p[1];

		/* Diagonal propagator */
		for (s=0; s<2; s++) al[stp][s] = ep[s];

		/* Upper & lower subdiagonal propagators */
		for (i=1; i<=NX; i++) { /* Loop over mesh points */
			if (stp==0) { /* Half-step */
				up = i%2;     /* Odd mesh point has upper off-diagonal */
				lw = (i+1)%2; /* Even               lower              */
			}
			else { /* Full step */
				up = (i+1)%2; /* Even mesh point has upper off-diagonal */
				lw = i%2;     /* Odd                 lower              */
			}
			for (s=0; s<2; s++) {
				bux[stp][i][s] = up*em[s];
				blx[stp][i][s] = lw*em[s];
			}
		} /* Endfor mesh points, i */
	} /* Endfor half & full steps, stp */

	/* Set up potential propagator */
	for (i=1; i<=NX; i++) {
		x = dx*i + LX*myid;
		/* Construct the edge potential */
		if ((myid == 0 && i==1) || (myid == nprocs-1 && i==NX))
			v[i] = EH;
		/* Construct the barrier potential */
		else if (0.5*(LX*nprocs-BW)<x && x<0.5*(LX*nprocs+BW))
			v[i] = BH;
		else
			v[i] = 0.0;
		/* Half-step potential propagator */
		u[i][0] = cos(-0.5*DT*v[i]);
		u[i][1] = sin(-0.5*DT*v[i]);
	}
}

/*------------------------------------------------------------------------------
	Initializes the wave function as a traveling Gaussian wave packet.
------------------------------------------------------------------------------*/
void init_wavefn() {
	int sx,s;
	double x,gauss,psisq,psisq_global,norm_fac;

	/* Calculate the the wave function value mesh point-by-point */
	for (sx=1; sx<=NX; sx++) {
		x = LX*myid+dx*sx-X0;
		gauss = exp(-0.25*x*x/(S0*S0));
		psi[sx][0] = gauss*cos(sqrt(2.0*E0)*x);
		psi[sx][1] = gauss*sin(sqrt(2.0*E0)*x);
	}

	/* Normalize the wave function */
	psisq=0.0;
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psisq += psi[sx][s]*psi[sx][s];
	psisq *= dx;
	MPI_Allreduce(&psisq,&psisq_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    norm_fac = 1.0/sqrt(psisq_global);
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<2; s++)
			psi[sx][s] *= norm_fac;
}

/*------------------------------------------------------------------------------
	Initializes the GPU arrays
------------------------------------------------------------------------------*/
void host2device(double *d1, double h2[NX+2][2]) {
  double *h1;
  h1 = (double *) malloc(sizeof(double)*2*(NX+2));
  for (int i=0; i<NX+2; i++) {
    for (int j=0; j<=1; j++) h1[2*i+j] = h2[i][j];
  }
  cudaMemcpy(d1,h1,sizeof(double)*2*(NX+2),cudaMemcpyHostToDevice);
}

void device2host(double h2[NX+2][2], double *d1) {
  double *h1;
  h1 = (double *) malloc(sizeof(double)*2*(NX+2));
  cudaMemcpy(h1,d1,sizeof(double)*2*(NX+2),cudaMemcpyDeviceToHost);
  for (int i=0; i<NX+2; i++) {
    for (int j=0; j<=1; j++) h2[i][j] = h1[2*i+j];
  }
}

void init_gpu (int myid) {
  cudaSetDevice(myid%2);
  cudaMalloc((void**) &dev_psi,  sizeof(double)*2*(NX+2));  /* host: psi[NX+2][2] <-> device: dev_psi[(NX+2)*2] */
  cudaMalloc((void**) &dev_wrk,  sizeof(double)*2*(NX+2));  /*       psi[i][0|1]  <->         dev_psi[2i|2i+1]  */
  cudaMalloc((void**) &dev_al0,  sizeof(double)*2);
  cudaMalloc((void**) &dev_al1,  sizeof(double)*2);
  cudaMalloc((void**) &dev_bux0, sizeof(double)*2*(NX+2));
  cudaMalloc((void**) &dev_bux1, sizeof(double)*2*(NX+2));
  cudaMalloc((void**) &dev_blx0, sizeof(double)*2*(NX+2));
  cudaMalloc((void**) &dev_blx1, sizeof(double)*2*(NX+2));
  cudaMalloc((void**) &dev_u,    sizeof(double)*2*(NX+2));

  cudaMemcpy(dev_al0,al[0],sizeof(double)*2,cudaMemcpyHostToDevice);
  cudaMemcpy(dev_al1,al[1],sizeof(double)*2,cudaMemcpyHostToDevice);
  host2device(dev_bux0,bux[0]);
  host2device(dev_bux1,bux[1]);
  host2device(dev_blx0,blx[0]);
  host2device(dev_blx1,blx[1]);
  host2device(dev_u,u);
}

/*------------------------------------------------------------------------------
	Propagates the electron wave function for a unit time step, DT.
------------------------------------------------------------------------------*/
void single_step() {
	pot_prop();  /* half step potential propagation */

	kin_prop(0); /* half step kinetic propagation   */
	kin_prop(1); /* full                            */
	kin_prop(0); /* half                            */

	pot_prop();  /* half step potential propagation */
}

/*------------------------------------------------------------------------------
	Potential propagator for a half time step, DT/2.
------------------------------------------------------------------------------*/
__global__ void gpu_pot_prop (double *psi, double *u) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
          /* [0,12]      192         [0,191] */
  int sx = tid + 1; /* mesh point [1,192*13=2496] */
  int s0 = 2*sx;
  int s1 = 2*sx + 1;
  double wr = u[s0]*psi[s0] - u[s1]*psi[s1];
  double wi = u[s0]*psi[s1] + u[s1]*psi[s0];
  psi[s0] = wr;
  psi[s1] = wi;
}

void pot_prop() {
  host2device(dev_psi,psi);
  gpu_pot_prop<<<dimGrid,dimBlock>>>(dev_psi,dev_u);
  device2host(psi,dev_psi);
}

/*------------------------------------------------------------------------------
	Kinetic propagation for t (=0 for DT/2--half; 1 for DT--full) step.
-------------------------------------------------------------------------------*/
__global__ void gpu_kin_prop (double *psi, double *wrk, double *al, double *blx, double *bux) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
          /* [0,12]      192         [0,191] */
  int sx = tid + 1; /* mesh point [1,192*13=2496] */
  int s0 = 2*sx;
  int s1 = 2*sx + 1;
  int l0 = 2*(sx-1);
  int l1 = 2*(sx-1)+1;
  int u0 = 2*(sx+1);
  int u1 = 2*(sx+1)+1;

  double wr = al[0]*psi[s0] - al[1]*psi[s1];
  double wi = al[0]*psi[s1] + al[1]*psi[s0];
  wr += blx[s0]*psi[l0] - blx[s1]*psi[l1];
  wi += blx[s0]*psi[l1] + blx[s1]*psi[l0];
  wr += bux[s0]*psi[u0] - bux[s1]*psi[u1];
  wi += bux[s0]*psi[u1] + bux[s1]*psi[u0];
  wrk[s0] = wr;
  wrk[s1] = wi;
}

__global__ void gpu_wrk2psi (double *psi, double *wrk) {
  int tid = blockIdx.x*blockDim.x + threadIdx.x;
  int sx = tid + 1; /* mesh point [1,192*13=2496] */
  int s0 = 2*sx;
  int s1 = 2*sx + 1;
  psi[s0] = wrk[s0];
  psi[s1] = wrk[s1];
}

void kin_prop(int t) {
  periodic_bc();
  host2device(dev_psi,psi);
  if (t == 0) {
    gpu_kin_prop<<<dimGrid,dimBlock>>>(dev_psi,dev_wrk,dev_al0,dev_blx0,dev_bux0);
  } else {
    gpu_kin_prop<<<dimGrid,dimBlock>>>(dev_psi,dev_wrk,dev_al1,dev_blx1,dev_bux1);
  }

  gpu_wrk2psi<<<dimGrid,dimBlock>>>(dev_psi,dev_wrk);
  device2host(psi,dev_psi);
}

/*------------------------------------------------------------------------------
	Applies the periodic boundary condition to wave function PSI, by copying
	the boundary values to the auxiliary array positions at the other ends.
------------------------------------------------------------------------------*/
void periodic_bc() {
	int plw = (myid-1+nprocs)%nprocs; /* Lower partner process */
	int pup = (myid+1       )%nprocs; /* Upper partner process */
	double dbuf[2], dbufr[2];
	MPI_Status status;

	/* Cache boundary wave function value at the lower end */
	dbuf[0] = psi[NX][0];  dbuf[1] = psi[NX][1];  /* dbuf[0:1] <- psi[NX][0:1]; */
	MPI_Send(dbuf, 2,MPI_DOUBLE,pup,10,MPI_COMM_WORLD); /* Send dbuf to pup; */
	MPI_Recv(dbufr,2,MPI_DOUBLE,plw,10,MPI_COMM_WORLD,&status); /* Receive dbufr from plw; */
	psi[0][0] = dbufr[0];  psi[0][1] = dbufr[1];  /* psi[0][0:1] <- dbufr[0:1]; */

	/* Cache boundary wave function value at the upper end */
	dbuf[0] = psi[1][0];  dbuf[1] = psi[1][1];  /* dbuf[0:1] <- psi[1][0:1]; */
	MPI_Send(dbuf, 2,MPI_DOUBLE,plw,20,MPI_COMM_WORLD); /* Send dbuf to plw; */
	MPI_Recv(dbufr,2,MPI_DOUBLE,pup,20,MPI_COMM_WORLD,&status); /* Receive dbufr from pup; */
	psi[NX+1][0] = dbufr[0];  psi[NX+1][1] = dbufr[1];  /* psi[NX+1][0:1] <- dbufr[0:1]; */
}

/*------------------------------------------------------------------------------
	Calculates the kinetic, potential & total energies, EKIN, EPOT & ETOT.
------------------------------------------------------------------------------*/
void calc_energy() {
	int sx,s;
	double a,bx;

	/* Apply the periodic boundary condition */
	periodic_bc();

	/* Tridiagonal kinetic-energy operators */
	a =   1.0/(dx*dx);
	bx = -0.5/(dx*dx);

	/* |WRK> = (-1/2)Laplacian|PSI> */
	for (sx=1; sx<=NX; sx++)
		for (s=0; s<=1; s++)
			wrk[sx][s] = a*psi[sx][s]+bx*(psi[sx-1][s]+psi[sx+1][s]);

	/* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
	ekin = 0.0;
	for (sx=1; sx<=NX; sx++)
		ekin += (psi[sx][0]*wrk[sx][0]+psi[sx][1]*wrk[sx][1]);
	ekin *= dx;
	MPI_Allreduce(&ekin,&ekin_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	/* Potential energy */
	epot = 0.0;
	for (sx=1; sx<=NX; sx++)
		epot += v[sx]*(psi[sx][0]*psi[sx][0]+psi[sx][1]*psi[sx][1]);
	epot *= dx;
	MPI_Allreduce(&epot,&epot_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);

	/* Total energy */
	etot = ekin+epot;
	MPI_Allreduce(&etot,&etot_global,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
}
