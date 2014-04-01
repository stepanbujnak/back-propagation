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

double *
nn_act_alloc(int n) {
  double *act;

  act = (double *)malloc(sizeof(double) * n);
  for (int i = 0; i < n; i++) {
    act[i] = 1.0;
  }
  
  return act;
}

void
nn_act_free(double *act) {
  free(act);
}

double **
nn_weight_alloc(int n, int m) {
  double **weight;

  srand(time(NULL));

  weight = (double **)malloc(sizeof(double *) * n);
  for (int i = 0; i < n; i++) {
    weight[i] = (double *)malloc(sizeof(double) * m);
    for (int j = 0; j < m; j++) {
      weight[i][j] = ((float)rand() / (float)(RAND_MAX / 2.0)) - 1.0;
    }
  }

  return weight;
}

void
nn_weight_free(double **w, int n) {
  for (int i = 0; i < n; i++) {
    free(w[i]);
  }
  free(w);
}

void
nn_init(struct nn *nn, int ni, int nh, int no) {
  nn->ni = ni;
  nn->nh = nh;
  nn->no = no;

  nn->ai = nn_act_alloc(ni);
  nn->ah = nn_act_alloc(nh);
  nn->ao = nn_act_alloc(no);

  nn->wh = nn_weight_alloc(ni, nh);
  nn->wo = nn_weight_alloc(nh, no);

  nn->ch = nn_weight_alloc(ni, nh);
  nn->co = nn_weight_alloc(nh, no);
}

void
nn_free(struct nn *nn) {
  nn_act_free(nn->ai);
  nn_act_free(nn->ah);
  nn_act_free(nn->ao);

  nn_weight_free(nn->wh, nn->ni);
  nn_weight_free(nn->wo, nn->nh);

  nn_weight_free(nn->ch, nn->ni);
  nn_weight_free(nn->co, nn->nh);
}

void
nn_update(struct nn *nn, int *inputs) {
  for (int i = 0; i < nn->ni; i++) {
    nn->ai[i] = inputs[i];
  }

  for (int i = 0; i < nn->nh; i++) {
    double sum = 0.0;

    for (int j = 0; j < nn->ni; j++) {
      sum += nn->ai[j] * nn->wh[j][i];
    }

    nn->ah[i] = tanh(sum);
  }

  for (int i = 0; i < nn->no; i++) {
    double sum = 0.0;

    for (int j = 0; j < nn->nh; j++) {
      sum += nn->ah[j] * nn->wo[j][i];
    }

    nn->ao[i] = tanh(sum);
  }
}

void
nn_back_propagate(struct nn *nn, int *targets) {
  double *ods, *hds;

  /* Output deltas */
  ods = (double *)malloc(sizeof(double) * nn->no);
  for (int i = 0; i < nn->no; i++) {
    double error;

    error = targets[i] - nn->ao[i];
    ods[i] = nn->ao[i] * (1 - nn->ao[i]) * error;
  }

  /* Hidden deltas */
  hds = (double *)malloc(sizeof(double) * nn->nh);
  for (int i = 0; i < nn->nh; i++) {
    double error = 0.0;

    for (int j = 0; j < nn->no; j++) {
      error += ods[j] * nn->wo[i][j];
    }

    hds[i] = nn->ah[i] * (1 - nn->ah[i]) * error;
  }

  /* Update output weights */
  for (int i = 0; i < nn->nh; i++) {
    for (int j = 0; j < nn->no; j++) {
      double change;
      
      change = ods[j] * nn->ah[i];
      nn->wo[i][j] += NN_RATE * change + NN_MOMENTUM * nn->co[i][j];
      nn->co[i][j] = change;
    }
  }

  /* Update hidden weights */
  for (int i = 0; i < nn->ni; i++) {
    for (int j = 0; j < nn->nh; j++) {
      double change;

      change = hds[j] * nn->ai[i];
      nn->wh[i][j] += NN_RATE * change + NN_MOMENTUM * nn->ch[i][j];
      nn->ch[i][j] = change;
    }
  }

  /* Cleanup */
  free(ods);
  free(hds);
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
