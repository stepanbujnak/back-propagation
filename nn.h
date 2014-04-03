/******************************************************************************
 * School of Engineering
 * Robert Gordon University, Aberdeen, UK
 ******************************************************************************
 * File name : nn.h
 * Author    : Stepan Bujnak (1111790)
 * Created   : 28/03/2014
 * License   : Public Domain
 ******************************************************************************
 * Title     : Artificial Neural Networks With Back Propagation
 * Module    : EN4541 Advanced Computer Architecture
 * Course    : Computer Science BSc. (Hons)
 * Year      : CM5
 ******************************************************************************
 * Notes     : Developed for C99 standard of C Programming Language
 ******************************************************************************/

#ifndef NN_H
#define NN_H

#define NN_ITERATIONS 1000000
#define NN_E 2.7182818284590452354
#define NN_RATE 1
#define NN_MOMENTUM 0.1

struct nn {
  /* Number of input, hidden and output nodes */
  int ni;
  int nh;
  int no;

  /* Activations for nodes */
  double *ai;
  double *ah;
  double *ao;

  /* Weight matrices */
  double **wh;
  double **wo;

  /* Last changes */
  double **ch;
  double **co;

  /* Deltas */
  double *hd; /* Hidden deltas */
  double *od; /* Output deltas */
};

int nn_init(struct nn *, int, int, int);
int nn_del(struct nn *);
void nn_train(struct nn *nn, int, int [][nn->ni], int [][nn->no]);
void nn_test(struct nn *nn, int, int [][nn->ni], int [][nn->no]);

#endif /* NN_H */
