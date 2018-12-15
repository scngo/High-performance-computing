/***********************************************************************
  Program atomv.c--ball representation of atoms.
  Required files
    atomv.h:   Include file
    md.conf:   MD configuration file containing atomic coordinates
  gcc animqd.c -o animqd -lm -lGL -lGLU -lglut
***********************************************************************/
#include "animqd.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include <float.h>
// #include <OpenGL/gl.h>          // Header File For The OpenGL32 Library
// #include <OpenGL/glu.h>         // Header File For The GLu32 Library
// #include <GLUT/glut.h>          // Header File For The GLut Library
#include <GL/glut.h>

GLuint sphereid;          /* display-list id of atom sphere geom */
GLuint atomsid;           /* display-list id of all atoms */
GLdouble fovy, aspect, near_clip, far_clip;  
                          /* parameters for gluPerspective() */

void main(int argc, char **argv) {
  /* Initialize QD */
  init_param();
  init_prop();
  init_wavefn();

  glutInit(&argc, argv);
  init_opengl();

  /* Set a glut callback functions */
  glutDisplayFunc(display);
  glutIdleFunc(idle);

  /* Start main display loop */
  glutMainLoop();
}

/*----------------------------------------------------------------------------*/
void init_param() {
/*------------------------------------------------------------------------------
  Initializes parameters.
------------------------------------------------------------------------------*/
  /* Read control parameters */
  FILE *input;
  input=fopen("qd.in","r");

  fscanf(input,"%le%le",&LX,&LY);
  fscanf(input,"%le",&DT);
  fscanf(input,"%d",&NSTEP);
  fscanf(input,"%d",&NECAL);
  fscanf(input,"%le%le%le",&X0,&S0,&E0);
  fscanf(input,"%le%le",&BH,&BW);
  fscanf(input,"%le",&EH);

  fclose(input);

  /* Calculate the mesh size */
  dx = LX/NX;
  dy = LY/NY;
}

/*----------------------------------------------------------------------------*/
void init_prop() {
/*------------------------------------------------------------------------------
  Initializes the kinetic & potential propagators.
------------------------------------------------------------------------------*/
  int dir,lim,stp,s,i,up,lw;
  int sx,sy;
  double a,exp_p[2],ep[2],em[2];
  double x,y;

  /* Set up kinetic propagators */
  for (dir=0; dir<ND; dir++) { /* Loop over x & y directions */
    if (dir==0) {
      lim=NX;
      a=0.5/(dx*dx);
    }
    else if (dir==1) {
      lim=NY;
      a=0.5/(dy*dy);
    }

    for (stp=0; stp<2; stp++) { /* Loop over half & full steps */
      exp_p[0]=cos(-(stp+1)*DT*a);
      exp_p[1]=sin(-(stp+1)*DT*a);
      ep[0]=0.5*(1.0+exp_p[0]);
      ep[1]=0.5*exp_p[1];
      em[0]=0.5*(1.0-exp_p[0]);
      em[1]= -0.5*exp_p[1];

      /* Diagonal propagator */
      for (s=0; s<2; s++) al[dir][stp][s]=ep[s];

      /* upper & lower subdiagonal propagators */
      for (i=1; i<=lim; i++) { /* Loop over mesh points */
        if (stp==0) { /* Half-step */
          up=i%2;     /* Odd mesh point has upper off-diagonal */
          lw=(i+1)%2; /* Even               lower              */
        }
        else { /* Full step */
          up=(i+1)%2; /* Even mesh point has upper off-diagonal */
          lw=i%2;     /* Odd                 lower              */
        }
        if (dir==0)
          for (s=0; s<2; s++) {
            bux[stp][i][s]=up*em[s];
            blx[stp][i][s]=lw*em[s];
          }
        else if (dir==1)
          for (s=0; s<2; s++) {
            buy[stp][i][s]=up*em[s];
            bly[stp][i][s]=lw*em[s];
          }
      } /* Endfor mesh points */
    } /* Endfor half & full steps */
  } /* Endfor x & y directions */

  /* Set up potential propagator */
  for (sx=1; sx<=NX; sx++) {
    x=dx*sx;
    for (sy=1; sy<=NY; sy++) {
      y=dy*sy;
      /* Construct the edge potential */
      if (sx==1 || sx==NX || sy==1 || sy==NY)
        v[sx][sy]=EH;
      /* Construct the barrier potential */
      else if (0.5*(LX-BW)<x && x<0.5*(LX+BW))
        v[sx][sy]=BH;
      else
        v[sx][sy]=0.0;
      /* Half-step potential propagator */
      u[sx][sy][0]=cos(-0.5*DT*v[sx][sy]);
      u[sx][sy][1]=sin(-0.5*DT*v[sx][sy]);
    }
  }
}

/*----------------------------------------------------------------------------*/
void init_wavefn() {
/*------------------------------------------------------------------------------
  Initializes the wave function as a Gaussian wave packet.
------------------------------------------------------------------------------*/
  int sx,sy,s;
  double x,y,gauss,psisq,norm_fac;
  char line[MAXLINE];

  FILE *f;
  f=fopen("Lenna512x512.pgm","r");

  for (s=1; s<=6; s++)
    fgets(line,MAXLINE,f);

  for (sy=NY; sy>=1; sy--) {
    for (sx=1; sx<=NX; sx++) {
      psi[sx][sy][0] = (double)fgetc(f);
    }
  }
  fclose(f);

  /* Calculate the value of the wave function mesh point-by-point */
  for (sx=1; sx<=NX; sx++) {
    x = dx*sx-X0;
    for (sy=1; sy<=NY; sy++) {
      if (sy==1 || sy==NY)
        gauss = 0.0;
      else {
        y = dy*sy;
        // gauss = exp(-0.25*x*x/(S0*S0))*sin(PI*(y-dy)/(LY-2.0*dy));
        gauss = psi[sx][sy][0];
      }
      psi[sx][sy][0] = gauss*cos(sqrt(2.0*E0)*x);
      psi[sx][sy][1] = gauss*sin(sqrt(2.0*E0)*x);
    }
  }

  /* Normalize the wave function */
  psisq=0.0;
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      for (s=0; s<=1; s++)
        psisq += psi[sx][sy][s]*psi[sx][sy][s];
  psisq *= dx*dy;
  norm_fac = 1.0/sqrt(psisq);
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      for (s=0; s<=1; s++)
        psi[sx][sy][s] *= norm_fac;
}

void init_opengl() {
  /* Set up an window */
  /* Initialize display mode */
  glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA|GLUT_DEPTH);
  /* Specify window size */
  glutInitWindowSize(winx, winy);
  /* Open window */
  glutCreateWindow("Quantum Dynamics Anumation");

  for (int i=0; i<NX; i++) {
    for (int j=0; j<NY; j++) {
      position[i][j][0] = (dx*i-0.5*LX)/LX;
      position[i][j][1] = (dy*j-0.5*LY)/LY;
    }
  }
  /* set initial eye & look at location in world space */
  center[0] = 0.0;
  center[1] = 0.0;
  center[2] = 0.0;
  eye[0] = center[0] + 0.1;
  eye[1] = center[1] + 0.1;
  eye[2] = center[2] + 1.0;
  up[0] = 0.0;
  up[1] = 1.0;
  up[2] = 0.0;
  gluLookAt(
    (GLdouble)eye[0],(GLdouble)eye[1],(GLdouble)eye[2],
    (GLdouble)center[0],(GLdouble)center[1],(GLdouble)center[2],
    (GLdouble)up[0],(GLdouble)up[1],(GLdouble)up[2]);
  glEnable(GL_COLOR_MATERIAL);
}

void display() {
  int i, j;
  double x, y, r;
  glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT);
  for (i=1; i<=NX; i++) {
    for (j=1; j<=NY; j++) {
      x = psi[i][j][0];
      y = psi[i][j][1];
      r = sqrt(x*x + y*y);
      position[i][j][2] = 10.0*r;
      rgbcolor[i][j][0] = fabs(x)/r;
      rgbcolor[i][j][1] = fabs(y)/r;
      rgbcolor[i][j][2] = 0.5;
    }
  }
  /* Quadrilateral stripes to cover the sphere */
  for (i=1; i<NX; i++) {
    glBegin(GL_QUAD_STRIP);
      for (j=0; j<=NY; j++) {
        glVertex3f(position[i][j][0],position[i][j][1],position[i][j][2]);
        glColor3f (rgbcolor[i][j][0],rgbcolor[i][j][1],rgbcolor[i][j][2]);
        glVertex3f(position[i+1][j][0],position[i+1][j][1],position[i+1][j][2]);
        glColor3f (rgbcolor[i+1][j][0],rgbcolor[i+1][j][1],rgbcolor[i+1][j][2]);
      }
    glEnd();
  }
  glutSwapBuffers();
}

void idle() {
  if (step<NSTEP) {
    single_step();
    if (step%NECAL==0) {
      calc_energy();
      printf("%le %le %le %le\n",DT*step,ekin,epot,etot);
    }
    step++;
    glutPostRedisplay();
  }
}

/*----------------------------------------------------------------------------*/
void single_step() {
/*------------------------------------------------------------------------------
  Propagates the electron wave function for a unit time step, DT.
------------------------------------------------------------------------------*/
  pot_prop();    /* half step potential propagation                  */

  kin_prop(0,0); /* half step kinetic propagation in the x direction */
  kin_prop(0,1); /* full                                             */
  kin_prop(0,0); /* half                                             */

  kin_prop(1,0); /* half step kinetic propagation in the y direction */
  kin_prop(1,1); /* full                                             */
  kin_prop(1,0); /* half                                             */

  pot_prop();    /* half step potential propagation                  */
}

/*----------------------------------------------------------------------------*/
void pot_prop() {
/*------------------------------------------------------------------------------
  Potential propagator for a half time step, DT/2.
------------------------------------------------------------------------------*/
  int sx,sy;
  double wr,wi;

  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++) {
      wr=u[sx][sy][0]*psi[sx][sy][0]-u[sx][sy][1]*psi[sx][sy][1];
      wi=u[sx][sy][0]*psi[sx][sy][1]+u[sx][sy][1]*psi[sx][sy][0];
      psi[sx][sy][0]=wr;
      psi[sx][sy][1]=wi;
    }
}

/*----------------------------------------------------------------------------*/
void kin_prop(int d, int t) {
/*------------------------------------------------------------------------------
  Kinetic propagation in the d (=0 for x; 1 for y) direction
  for t (=0 for DT/2--half; 1 for DT--full) step.
-------------------------------------------------------------------------------*/
  int sx,sy,s;
  double wr,wi;

  /* Apply the periodic boundary condition */
  periodic_bc();

  /* WRK|PSI holds the new|old wave function */
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++) {
      wr=al[d][t][0]*psi[sx][sy][0]-al[d][t][1]*psi[sx][sy][1];
      wi=al[d][t][0]*psi[sx][sy][1]+al[d][t][1]*psi[sx][sy][0];
      if (d==0) {
        wr+=(blx[t][sx][0]*psi[sx-1][sy][0]-blx[t][sx][1]*psi[sx-1][sy][1]);
        wi+=(blx[t][sx][0]*psi[sx-1][sy][1]+blx[t][sx][1]*psi[sx-1][sy][0]);
        wr+=(bux[t][sx][0]*psi[sx+1][sy][0]-bux[t][sx][1]*psi[sx+1][sy][1]);
        wi+=(bux[t][sx][0]*psi[sx+1][sy][1]+bux[t][sx][1]*psi[sx+1][sy][0]);
      }
      else if (d==1) {
        wr+=(bly[t][sy][0]*psi[sx][sy-1][0]-bly[t][sy][1]*psi[sx][sy-1][1]);
        wi+=(bly[t][sy][0]*psi[sx][sy-1][1]+bly[t][sy][1]*psi[sx][sy-1][0]);
        wr+=(buy[t][sy][0]*psi[sx][sy+1][0]-buy[t][sy][1]*psi[sx][sy+1][1]);
        wi+=(buy[t][sy][0]*psi[sx][sy+1][1]+buy[t][sy][1]*psi[sx][sy+1][0]);
      }
      wrk[sx][sy][0]=wr;
      wrk[sx][sy][1]=wi;
    }

  /* Copy the new wave function back to PSI */
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      for (s=0; s<=1; s++)
        psi[sx][sy][s]=wrk[sx][sy][s];
}

/*----------------------------------------------------------------------------*/
void periodic_bc() {
/*------------------------------------------------------------------------------
  Applies the periodic boundary condition to wave function PSI.
------------------------------------------------------------------------------*/
  int sx,sy,s;

  /* Copy boundary wave function values in the x direction */
  for (sy=1; sy<=NY; sy++)
    for (s=0; s<=1; s++) {
      psi[0][sy][s] = psi[NX][sy][s];
      psi[NX+1][sy][s] = psi[1][sy][s];
    }

  /* Copy boundary wave function values in the y direction */
  for (sx=1; sx<=NX; sx++)
    for (s=0; s<=1; s++) {
      psi[sx][0][s] = psi[sx][NY][s];
      psi[sx][NY+1][s] = psi[sx][1][s];
    }
}

/*----------------------------------------------------------------------------*/
void calc_energy() {
/*------------------------------------------------------------------------------
  Calculates the kinetic, potential & total energies.
------------------------------------------------------------------------------*/
  int sx,sy,s;
  double a,bx,by;

  /* Apply the periodic boundary condition */
  periodic_bc();

  /* Tridiagonal kinetic-energy operators */
  a=1.0/(dx*dx)+1.0/(dy*dy);
  bx= -0.5/(dx*dx);
  by= -0.5/(dy*dy);

  /* |WRK> = (-1/2)Laplacian|PSI> */
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      for (s=0; s<=1; s++)
        wrk[sx][sy][s] = a*psi[sx][sy][s]
                       + bx*(psi[sx-1][sy][s]+psi[sx+1][sy][s])
                       + by*(psi[sx][sy-1][s]+psi[sx][sy+1][s]);

  /* Kinetic energy = <PSI|(-1/2)Laplacian|PSI> = <PSI|WRK> */
  ekin = 0.0;
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      ekin += (psi[sx][sy][0]*wrk[sx][sy][0]+psi[sx][sy][1]*wrk[sx][sy][1]);
  ekin *= dx*dy;

  /* Potential energy */
  epot = 0.0;
  for (sx=1; sx<=NX; sx++)
    for (sy=1; sy<=NY; sy++)
      epot += v[sx][sy]*(psi[sx][sy][0]*psi[sx][sy][0]+psi[sx][sy][1]*psi[sx][sy][1]);
  epot *= dx*dy;

  /* Total energy */
  etot = ekin+epot;
}