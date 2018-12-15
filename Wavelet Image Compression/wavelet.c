/* g++ wavelet.c -o wavelet */
#include <stdio.h>

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
double div[4][N+2][N+2];
int vproc[2] = {2,2}; /* 2-by-2 spatial decomposition */
int nproc = 4; /* vproc[0]*vproc[1] number of procs */
int vid[2]; /* Vector process ID */
int nbr[2]; /* Neighbor id of rows and columns */
int nr, nc; /* Num of rows & columns per processor */

int read_img();
int wavelet();
int write_img();

int main(int argc, char *argv[]) {
  nr = N/vproc[0];
  nc = N/vproc[1];
  read_img(); /* Read imput image */

  /* Recursive wavelet transform */
  for (int l=0; l<L; l++) {
    wavelet();
    nr /= 2;
    nc /= 2;
  }

  write_img(); /* Output image */
  return 0;
}

int read_img() {
  FILE *input;
  char line[MAXLINE];
  int r, c, s, sr, sc;
  double buf[N*N];

  input = fopen("Lenna512x512.pgm","r");
  fgets(line,MAXLINE,input);
  fgets(line,MAXLINE,input);
  fgets(line,MAXLINE,input);
  fgets(line,MAXLINE,input);
  for (r=0; r<N; r++) for (c=0; c<N; c++) img[r][c] = (int) fgetc(input);
  fclose(input);
  for (s=0; s<nproc; s++) {
    sr = s/vproc[1];
    sc = s%vproc[1];
    for (r=sr*nr; r<(sr+1)*nr; r++) for (c=sc*nc; c<(sc+1)*nc; c++) buf[nc*(r-sr*nr)+c-sc*nc] = img[r][c];
    for (r=0; r<nr; r++) for (c=0; c<nc; c++) div[s][r][c] = buf[nc*r+c];
  }
  for (r=0; r<N; r++) for (c=0; c<N; c++) img[r][c] = 0.0;
  return 0;
}

int wavelet() {
  int c, r, s, s0, s1;
  double buf[N*N];
  for (s=0; s<nproc; s++) {
    vid[0] = s/vproc[1];
    vid[1] = s%vproc[1];
    s0 = (vid[0]-1+vproc[0])%vproc[0];
    s1 = vid[1];
    nbr[0] = s0 * vproc[1] + s1;
    s0 = vid[0];
    s1 = (vid[1]-1+vproc[1])%vproc[1];
    nbr[1] = s0 * vproc[1] + s1;
    for (r=0; r<=1; r++) for (c=0; c<nc; c++) buf[nc*r+c] = div[s][r][c]; 
    for (r=0; r<=1; r++) for (c=0; c<nc; c++) div[nbr[0]][nr+r][c] = buf[nc*r+c]; 
    for (c=0; c<=1; c++) for (r=0; r<nr/2; r++) buf[nr*c+r] = div[s][r][c]; 
    for (c=0; c<=1; c++) for (r=0; r<nr/2; r++) div[nbr[1]][r][nc+c] = buf[nr*c+r];
  }
  for (s=0; s<nproc; s++) {
    for (c=0; c<nc; c++) for (r=0; r<nr/2; r++) {
        div[s][r][c] = C0*div[s][2*r][c] + C1*div[s][2*r+1][c] + C2*div[s][2*r+2][c] + C3*div[s][2*r+3][c];
        div[s][r][c] /= SQRT2;
      }
    for (r=0; r<nr/2; r++) for (c=0; c<nc/2; c++) {
        div[s][r][c] = C0*div[s][r][2*c] + C1*div[s][r][2*c+1] + C2*div[s][r][2*c+2] + C3*div[s][r][2*c+3];
        div[s][r][c] /= SQRT2;
    }
  }
  return 0;
}

int write_img() {
  FILE *output;
  int c, r, s, sr, sc;
  double buf[N*N];
  for (r=0; r<N; r++) for (c=0; c<N; c++) img[r][c] = 0.0;
  for (s=0; s<nproc; s++) {
    sr = s/vproc[1];
    sc = s%vproc[1];
    for (r=0; r<nr; r++) for (c=0; c<nc; c++) buf[nc*r+c] = div[s][r][c];
    for (r=sr*nr; r<(sr+1)*nr; r++) for (c=sc*nc; c<(sc+1)*nc; c++) img[r][c] = buf[nc*(r-sr*nr)+c-sc*nc];
  }
  output = fopen("Lenna64x64.pgm","w");
  fprintf(output,"P5\n");
  fprintf(output,"# Simple image test\n");
  fprintf(output,"%d %d\n",nc*vproc[1],nr*vproc[0]);
  fprintf(output,"%d\n",MAX);
  for (r=0; r<nr*vproc[0]; r++) for (c=0; c<nc*vproc[1]; c++) fputc((char) img[r][c],output);
  fclose(output);
  return 0;
}