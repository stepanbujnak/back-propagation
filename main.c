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

int
main(int argc, char *argv[]) {
  struct nn nn;
  int inputs[][2] = {
    {0, 0},
    {1, 0},
    {0, 1},
    {1, 1}
  };
  int targets[][1] = {
    {0},
    {1},
    {1},
    {0}
  };

  nn_init(&nn, 2, 2, 1);
  nn_train(&nn, 4, inputs, targets);
  nn_test(&nn, 4, inputs, targets);
  nn_free(&nn);

  return 0;
}
