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

#define NN_ITERATIONS 10000
#define NN_RATE 0.5
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
};

double *nn_act_alloc(int);
void nn_act_free(double *);
double **nn_weight_alloc(int, int);
void nn_weight_free(double **, int);
void nn_init(struct nn *, int, int, int);
void nn_free(struct nn *);
void nn_update(struct nn *, int *);
void nn_back_propagate(struct nn *, int *);
void nn_train(struct nn *nn, int, int [][nn->ni], int [][nn->no]);
void nn_test(struct nn *nn, int, int [][nn->ni], int [][nn->no]);

#endif /* NN_H */
