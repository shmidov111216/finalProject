#ifndef SYMNMF_H
#define SYMNMF_H
#define INIT_CAP 32
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "matrix_util.h"

#ifdef PYTHON_BUILD
    #define FROM_PYTHON 1
#else
    #define FROM_PYTHON 0
#endif


#define CHECK_FREE_AND_EXIT(ptr)                     \
    do                                               \
    {                                                \
        if (!(ptr))                                  \
        {                                            \
            if (!FROM_PYTHON)                        \
            {                                        \
                printf("An Error Has Occurred C\n"); \
            }                                        \
            pool_free_all(MAIN_POOL);                \
            pool_free_all(TEMP_POOL);                \
            return 0;                                \
        }                                            \
    } while (0)

MatrixPtr updateH(MatrixPtr H, MatrixPtr W);
MatrixPtr getResultH(MatrixPtr H, MatrixPtr W);
MatrixPtr getSimilarityMatrix(MatrixPtr X);
MatrixPtr getDiagonalDegreeMatrix(MatrixPtr A);
MatrixPtr getNormalizedSimilarityMatrix(MatrixPtr A, MatrixPtr D);

#endif