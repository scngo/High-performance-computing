/* mpicc -fopenmp mpwavelet.c -o mpwavelet */
#include <stdio.h>
#include <mpi.h>
#include <omp.h>

#define N 512 /* Number of pixels in original image */
#define L 3 /* Number of image halvings */
#define MAX 255
#define MAXLINE 1024

#define C0  0.4829629131445341 /* (1+√3)/4√2 */
#define C1  0.8365163037378079 /* (3+√3)/4√2 */
#define C2  0.2241438680420134 /* (3-√3)/4√2 */
#define C3 -0.1294095225512604 /* (1-√3)/4√2 */
#define SQRT2 1.41421356237 /* √2 */

double img[N+2][N+2]; /* To append 2 rows & 2 columns */
int vproc[2] = {2,2}; /* 2-by-2 spatial decomposition */
int nproc = 4; /* vproc[0]*vproc[1] number of procs */
int vid[2]; /* Vector process ID */
int sid; /* Serial process ID */
int nbr[2]; /* Neighbor id of rows and columns */
int nr, nc; /* Num of rows & columns per processor */
int nthread = 4; /* Num of OpenMP thread */

int read_img();
int wavelet();
int write_img();

int main(int argc, char *argv[]) {
  MPI_Init(&argc,&argv);
  int s0, s1, l;
  MPI_Comm_rank(MPI_COMM_WORLD,&sid);
  vid[0] = sid/vproc[1];
  vid[1] = sid%vproc[1];

  s0 = (vid[0]-1+vproc[0])%vproc[0];
  s1 = vid[1];
  nbr[0] = s0 * vproc[1] + s1;
  s0 = vid[0];
  s1 = (vid[1]-1+vproc[1])%vproc[1];
  nbr[1] = s0 * vproc[1] + s1;
  nr = N/vproc[0]; /* Num of rows and colums of pixels I am in charge */
  nc = N/vproc[1];

  omp_set_num_threads(nthread); /* OpenMP setup 4 */
  read_img(); /* Read imput image */

  for (l=0; l<L; l++) { /* Recursive wavelet transform */
    wavelet();
    nr /= 2;
    nc /= 2;
  }

  write_img(); /* Output image */
  MPI_Finalize();
  return 0;
}

int read_img() {
  FILE *input;
  char line[MAXLINE];
  int r, c, s, sr, sc;
  double buf[N*N];
  MPI_Status status;
  if (sid == 0) { /* read img[0:N-1][0:N-1] */
    input = fopen("Lenna512x512.pgm","r");
    fgets(line,MAXLINE,input);
    fgets(line,MAXLINE,input);
    fgets(line,MAXLINE,input);
    fgets(line,MAXLINE,input);
    for (r=0; r<N; r++) for (c=0; c<N; c++)
      	img[r][c] = (int) fgetc(input);
    fclose(input);
  }
  for (s=1; s<nproc; s++) {
    sr = s/vproc[1];
    sc = s%vproc[1];
    if (sid == 0) { /* send image[sr*nr:(sr+1)*nr-1][sc*nc:(sc+1)*nc-1] to rank s */
      for (r=sr*nr; r<(sr+1)*nr; r++) for (c=sc*nc; c<(sc+1)*nc; c++) buf[nc*(r-sr*nr)+c-sc*nc] = img[r][c];
      MPI_Send(buf,nr*nc,MPI_DOUBLE,s,10,MPI_COMM_WORLD); 
    } else if (sid == s) { /* recv image[0:nr-1][0:nc-1] from rank 0 */
      MPI_Recv(buf,nr*nc,MPI_DOUBLE,0,10,MPI_COMM_WORLD,&status);
      for (r=0; r<nr; r++) for (c=0; c<nc; c++) img[r][c] = buf[nc*r+c];
    }
  }

  return 0;
}

int wavelet() {
  int r, c;
  double rbuf[2*N], sbuf[2*N];
  MPI_Status status;
  MPI_Request request;

  /* Cache 2 rows: Send & recv buffer size (2*nc) */
  /* (a) Compose a message: sbuf[0:2*nc-1] <- img[0:1][0:nc-1] */
  for (r=0; r<=1; r++) for (c=0; c<nc; c++) sbuf[nc*r+c] = img[r][c];
  /* (b) Do asynchronous message passing: */
  MPI_Irecv(rbuf,2*nc,MPI_DOUBLE,MPI_ANY_SOURCE,20,MPI_COMM_WORLD,&request);
  MPI_Send(sbuf,2*nc,MPI_DOUBLE,nbr[0],20,MPI_COMM_WORLD);
  MPI_Wait(&request,&status);
  /* (c) Message append: img[nr:nr+1][0:nc-1] <- rbuf[0:2*nc-1] */
  for (r=0; r<=1; r++) for (c=0; c<nc; c++) img[nr+r][c] = rbuf[nc*r+c];

  /* Row wavelet transform */
#pragma omp parallel for private(r) /* c is privatized automatically */
  for (c=0; c<nc; c++) for (r=0; r<nr/2; r++) {
      img[r][c] = C0*img[2*r][c] + C1*img[2*r+1][c] + C2*img[2*r+2][c] + C3*img[2*r+3][c];
      img[r][c] /= SQRT2;
    }

  /* Cache 2 columns */
  for (c=0; c<=1; c++) for (r=0; r<nr/2; r++) sbuf[nr*c/2+r] = img[r][c];
  /* Irecv img[0:nr/2-1][nc:nc+1] from MPI_ANY_SOURSE */
  MPI_Irecv(rbuf,nr,MPI_DOUBLE,MPI_ANY_SOURCE,30,MPI_COMM_WORLD,&request);
  /* Send img[0:nr/2-1][0:1] to nbr[1] */
  MPI_Send(sbuf,nr,MPI_DOUBLE,nbr[1],30,MPI_COMM_WORLD);
  MPI_Wait(&request,&status);
  for (c=0; c<=1; c++) for (r=0; r<nr/2; r++) img[r][nc+c] = rbuf[nr*c/2+r];

  /* Column wavelet transform */
#pragma omp parallel for private(c) /* r is privatized automatically */
  for (r=0; r<nr/2; r++) for (c=0; c<nc/2; c++) {
      img[r][c] = C0*img[r][2*c] + C1*img[r][2*c+1] + C2*img[r][2*c+2] + C3*img[r][2*c+3];
      img[r][c] /= SQRT2;
    }

  return 0;
}

int write_img() {
  FILE *output;
  int r, c, s, sr, sc;
  double buf[N*N];
  MPI_Status status;
  for (s=1; s<nproc; s++) {
    sr = s/vproc[1];
    sc = s%vproc[1];
    if (sid == s) { /* send img[0:nr-1][0:nc-1] to rank 0 */
      for (r=0; r<nr; r++) for (c=0; c<nc; c++) buf[nc*r+c] = img[r][c];
      MPI_Send(buf,nr*nc,MPI_DOUBLE,0,40,MPI_COMM_WORLD);
    } else if (sid == 0) { /* recv img[sr*nr:(sr+1)*nr-1][sc*nc:(sc+1)*nc-1] from rank s */
      MPI_Recv(buf,nr*nc,MPI_DOUBLE,s,40,MPI_COMM_WORLD,&status);
      for (r=sr*nr; r<(sr+1)*nr; r++) for (c=sc*nc; c<(sc+1)*nc; c++) img[r][c] = buf[nc*(r-sr*nr)+c-sc*nc];
    }
  }
  if (sid == 0) {
    /* rescale img[][] to [0,255] */
    /* write img[0:nr*vproc[0]-1][0:nc*vproc[1]-1] */
    output = fopen("Lenna64x64.pgm","w");
    fprintf(output,"P5\n");
    fprintf(output,"# Simple image test\n");
    fprintf(output,"%d %d\n",nc*vproc[1],nr*vproc[0]);
    fprintf(output,"%d\n",MAX);
    for (r=0; r<nr*vproc[0]; r++) for (c=0; c<nc*vproc[1]; c++)
        fputc((char) img[r][c],output);
    fclose(output);
  }
  return 0;
}
