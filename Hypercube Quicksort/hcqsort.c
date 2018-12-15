/* Hypercube Quicksort ********************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

#define N 1024 /* Maximum list size */
#define MAX 99 /* Maximum value of a list element */
#define MAXD 99
#define MAXP 99

int nprocs,dim,myid,pivot; /* Cube size, dimension, & my node ID */

void quicksort(int list[],int left,int right) {
  int pivot,i,j;
  int temp;
  if (left < right) {
    i = left;
    j = right + 1;
    pivot = list[left];
    do {
      while (list[++i] < pivot && i <= right);
      while (list[--j] > pivot);
      if (i < j) {
        temp = list[i];
        list[i] = list[j];
        list[j] = temp;
      }
    } while (i < j);
    temp = list[left];
    list[left] = list[j];
    list[j] = temp;
    quicksort(list,left,j-1);
    quicksort(list,j+1,right);
  }
}

int parallel_qsort(int myid,int list[],int n) {
  int bitvalue, mask, partner, nsend, nrecv, temp, nprocs_cube = nprocs;
  int i, j, l, c, p;
  int procs_cube[MAXP];
  MPI_Comm cube[MAXD][MAXP];
  MPI_Status status;
  MPI_Group cube_group[MAXD][MAXP];

  bitvalue = nprocs >> 1;
  mask = nprocs - 1;

  cube[dim][0] = MPI_COMM_WORLD;

  for (l = dim; l >= 1; l--) {
    if ((myid & mask) == 0) {
      for (i = 0, pivot = 0; i < n; i++) pivot += list[i];
      if (n > 0) pivot /= n;
    } 
    nprocs_cube = pow(2,l);
    c = myid/nprocs_cube; /* My subcube */
    MPI_Bcast(&pivot,1,MPI_INT,0,cube[l][c]);

    /* Partition list[0:nelement-1] into two sublists such that list[0:j] ≤ pivot < list[j+1:nelement-1]; */
    if (n > 0) {
      i = -1;
      j = n;
      do {
        while (list[++i] <= pivot && i < n);
        while (list[--j] > pivot && j > -1);
        if (i < j) {
          temp = list[i];
          list[i] = list[j];
          list[j] = temp;
        }
      } while (i < j);
    } else j = -1;

    /* Exchange the lower & upper sublists with the partner */
    partner = myid ^ bitvalue;
    if ((myid & bitvalue) == 0) { /* Lower */
      nsend = (n-j-1 >= 0) ? (n-j-1) : 0;
      MPI_Send(&nsend,1,MPI_INT,partner,10+l,MPI_COMM_WORLD); /* send the right sublist list[j+1:n] to partner */
      MPI_Recv(&nrecv,1,MPI_INT,partner,10+l,MPI_COMM_WORLD,&status); /* receive the right sublist of partner */
      MPI_Send(&(list[j+1]),nsend,MPI_INT,partner,20+l,MPI_COMM_WORLD);
      n -= nsend;
      MPI_Recv(&(list[n  ]),nrecv,MPI_INT,partner,20+l,MPI_COMM_WORLD,&status); /* append the received list to my right list */
      n += nrecv;
    } else { /* Upper */
      nsend = (j+1 >= 0) ? (j+1) : 0;
      MPI_Send(&nsend,1,MPI_INT,partner,10+l,MPI_COMM_WORLD); /* send the right sublist list[j+1:nelement-1] to partner */
      MPI_Recv(&nrecv,1,MPI_INT,partner,10+l,MPI_COMM_WORLD,&status); /* receive the left sublist of partner */
      MPI_Send(list,nsend,MPI_INT,partner,20+l,MPI_COMM_WORLD);
      if (nsend > 0) for (i = j+1; i < n; i++) list[i-j-1] = list[i];
      n -= nsend;
      MPI_Recv(&(list[n]),nrecv,MPI_INT,partner,20+l,MPI_COMM_WORLD,&status); /* append the received list to my left list */
      n += nrecv;
    }

    mask = mask ^ bitvalue;
    bitvalue = bitvalue >> 1;

    /* Split the world into two (l-1)-dimensional subcubes */
    MPI_Comm_group(cube[l][c],&(cube_group[l][c]));
    nprocs_cube /=2;
    for (p = 0; p < nprocs_cube; p++) procs_cube[p] = p;
    MPI_Group_incl(cube_group[l][c],nprocs_cube,procs_cube,&(cube_group[l-1][2*c  ]));
    MPI_Group_excl(cube_group[l][c],nprocs_cube,procs_cube,&(cube_group[l-1][2*c+1]));
    MPI_Comm_create(cube[l][c],cube_group[l-1][2*c  ],&(cube[l-1][2*c ]));
    MPI_Comm_create(cube[l][c],cube_group[l-1][2*c+1],&(cube[l-1][2*c+1]));
    MPI_Group_free(&(cube_group[l  ][c    ]));
    MPI_Group_free(&(cube_group[l-1][2*c  ]));
    MPI_Group_free(&(cube_group[l-1][2*c+1]));
    // MPI_Comm_Split(cube,cid/nprocs_cube,myid,&cube);
  }

  /* sequential quicksort to list[0:nelement-1] */
  quicksort(list,0,n-1);

  return n;
}


int main(int argc, char *argv[]) {
  int list[N],n=32,i;

  MPI_Init(&argc,&argv); /* Initialize the MPI environment */

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  dim = log2(nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD,&myid);

  srand((unsigned) myid+1);
  for (i=0; i<n/nprocs; i++) list[i] = rand()%MAX;

  printf("Before: Node %2d :",myid);
  for (i=0; i<n/nprocs; i++) printf("%3d ",list[i]);
  printf("\n");

  n = parallel_qsort(myid,list,n/nprocs);

  printf("After:  Node %2d :",myid);
  for (i=0; i<n; i++) printf("%3d ",list[i]);
  printf("\n");

  MPI_Finalize();

  return 0;
}
