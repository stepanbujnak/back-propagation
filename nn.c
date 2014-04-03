/******************************************************************************
 * School of Engineering
 * Robert Gordon University, Aberdeen, UK
 ******************************************************************************
 * File name : nn.c
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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "nn.h"

static void
nn_freep(void *arg) {
  void **ptr = (void **)arg;
  if (*ptr) {
    free(*ptr);
    *ptr = NULL;
  }
}

static double *
nn_act_alloc(int n) {
  double *act;

  act = (double *)malloc(sizeof(double) * n);
  if (!act) {
    return NULL;
  }

  for (int i = 0; i < n; i++) {
    act[i] = 1.0;
  }
  
  return act;
}

static double **
nn_weight_alloc(int n, int m) {
  double **weight;
  double *heap;

  /* Allocate continuously */
  weight = (double **)malloc(sizeof(double *) * n);
  if (!weight) {
    return NULL;
  }

  heap = (double *)malloc(sizeof(double) * n * m);
  if (!heap) {
    nn_freep(&weight);
    return NULL;
  }

  /* Fill with random numbers */
  for(int i = 0; i < n; i++) {
    weight[i] = heap + i * n;
    for (int j = 0; j < m; j++) {
      weight[i][j] = ((float)rand() / (float)(RAND_MAX / 2.0)) - 1.0;
    }
  }

  return weight;
}

static double
nn_sigmoid(double x) {
  return 1.0 / (1.0 + pow(NN_E, -x));
}

static void
nn_update(struct nn *nn, int *inputs) {
  for (int i = 0; i < nn->ni; i++) {
    nn->ai[i] = inputs[i];
  }

  for (int i = 0; i < nn->nh; i++) {
    double sum = 0.0;

    for (int j = 0; j < nn->ni; j++) {
      sum += nn->ai[j] * nn->wh[j][i];
    }

    nn->ah[i] = nn_sigmoid(sum);
  }

  for (int i = 0; i < nn->no; i++) {
    double sum = 0.0;

    for (int j = 0; j < nn->nh; j++) {
      sum += nn->ah[j] * nn->wo[j][i];
    }

    nn->ao[i] = nn_sigmoid(sum);
  }
}

static void
nn_back_propagate(struct nn *nn, int *targets) {
  /* Output deltas */
  for (int i = 0; i < nn->no; i++) {
    double error;

    error = targets[i] - nn->ao[i];
    nn->od[i] = nn->ao[i] * (1 - nn->ao[i]) * error;
  }

  /* Hidden deltas */
  for (int i = 0; i < nn->nh; i++) {
    double error = 0.0;

    for (int j = 0; j < nn->no; j++) {
      error += nn->od[j] * nn->wo[i][j];
    }

    nn->hd[i] = nn->ah[i] * (1 - nn->ah[i]) * error;
  }

  /* Update output weights */
  for (int i = 0; i < nn->nh; i++) {
    for (int j = 0; j < nn->no; j++) {
      double change;
      
      change = nn->od[j] * nn->ah[i];
      nn->wo[i][j] += NN_RATE * change + NN_MOMENTUM * nn->co[i][j];
      nn->co[i][j] = change;
    }
  }

  /* Update hidden weights */
  for (int i = 0; i < nn->ni; i++) {
    for (int j = 0; j < nn->nh; j++) {
      double change;

      change = nn->hd[j] * nn->ai[i];
      nn->wh[i][j] += NN_RATE * change + NN_MOMENTUM * nn->ch[i][j];
      nn->ch[i][j] = change;
    }
  }
}

int
nn_init(struct nn *nn, int ni, int nh, int no) {
  nn->ni = ni;
  nn->nh = nh;
  nn->no = no;

  /* Assign NULL so that nn_freep won't fail */
  nn->ai = NULL;
  nn->ah = NULL;
  nn->ao = NULL;
  nn->wh = NULL;
  nn->wo = NULL;
  nn->ch = NULL;
  nn->co = NULL;
  nn->hd = NULL;
  nn->od = NULL;

  nn->ai = nn_act_alloc(ni);
  if (!nn->ai) {
    return nn_del(nn);
  }

  nn->ah = nn_act_alloc(nh);
  if (!nn->ah) {
    return nn_del(nn);
  }

  nn->ao = nn_act_alloc(no);
  if (!nn->ao) {
    return nn_del(nn);
  }

  nn->hd = nn_act_alloc(nh);
  if (!nn->hd) {
    return nn_del(nn);
  }

  nn->od = nn_act_alloc(no);
  if (!nn->od) {
    return nn_del(nn);
  }

  nn->wh = nn_weight_alloc(ni, nh);
  if (!nn->wh) {
    return nn_del(nn);
  }

  nn->wo = nn_weight_alloc(nh, no);
  if (!nn->wo) {
    return nn_del(nn);
  }

  nn->ch = nn_weight_alloc(ni, nh);
  if (!nn->ch) {
    return nn_del(nn);
  }

  nn->co = nn_weight_alloc(nh, no);
  if (!nn->co) {
    return nn_del(nn);
  }

  return 0;
}

int
nn_del(struct nn *nn) {
  nn_freep(&nn->ai);
  nn_freep(&nn->ah);
  nn_freep(&nn->ao);
  nn_freep(&nn->hd);
  nn_freep(&nn->od);

  if (nn->wh) {
    nn_freep(&nn->wh[0]);
    nn_freep(&nn->wh);
  }

  if (nn->wo) {
    nn_freep(&nn->wo[0]);
    nn_freep(&nn->wo);
  }

  if (nn->ch) {
    nn_freep(&nn->ch[0]);
    nn_freep(&nn->ch);
  }

  if (nn->co) {
    nn_freep(&nn->co[0]);
    nn_freep(&nn->co);
  }

  return 1;
}

void
nn_train(struct nn *nn, int n, int inputs[][nn->ni], int targets[][nn->no]) {
  for (int i = 0; i < NN_ITERATIONS; i++) {
    for (int j = 0; j < n; j++) {
      nn_update(nn, inputs[j]);
      nn_back_propagate(nn, targets[j]);
    }
  }
}

void
nn_test(struct nn *nn, int n, int inputs[][nn->ni], int targets[][nn->no]) {
  for (int i = 0; i < n; i++) {
    nn_update(nn, inputs[i]);

    /* Print inputs */
    printf("%.2f", nn->ai[0]);
    for (int j = 1; j < nn->ni; j++) {
      printf(", %.2f", nn->ai[j]);
    }

    /* Print delimiter */
    printf(" -> ");

    /* Print outputs */
    printf("%.2f", nn->ao[0]);
    for (int j = 1; j < nn->no; j++) {
      printf(", %.2f", nn->ao[j]);
    }

    /* Print new line */
    printf("\n");
  }
}
