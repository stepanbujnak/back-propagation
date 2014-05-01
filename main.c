/******************************************************************************
 * School of Engineering
 * Robert Gordon University, Aberdeen, UK
 ******************************************************************************
 * File name : main.c
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

#include "nn.h"

#include <stdio.h>

int
main(int argc, char *argv[]) {
  struct nn nn;
  int inputs[][2] = {
    {1, 1},
    {0, 0}
  };
  int targets[][2] = {
    {1, 0},
    {0, 1}
  };

  if (nn_init(&nn, 2, 2, 2)) {
    fprintf(stderr, "The NN library could not be initialized\n");
  }

  nn_train(&nn, 2, inputs, targets);
  nn_test(&nn, 2, inputs, targets);
  nn_del(&nn);

  return 0;
}
