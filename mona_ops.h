/*
 * =====================================================================================
 *
 *       Filename:  mona_ops.h
 *
 *    Description:  For each given specific structure of MAT and VEC
 *                  User should give specific operations in MONA_OPS ops 
 *                  Operations: Mat 
 *                              Vec
 *                              Orth
 *                              Multigrid
 *                              LinearSolver
 *                              EigenSolver
 *                              Multivec
 *
 *        Version:  1.0
 *        Created:  2018年09月24日 09时50分16秒
 *       Revision:  none
 *       Compiler:  gcc
 *
 *         Author:  Li Yu (liyu@tjufe.edu.cn), 
 *   Organization:  TJUFE
 *
 * =====================================================================================
 */

#ifndef  _MONA_OPS_H_
#define  _MONA_OPS_H_

#include "mona_config.h"
#if MONA_USE_MPI
#include <mpi.h>
#endif

typedef struct MONA_OPS_ {

   /* REQUIRED Vec operations*/
   /* Create des_vec by src_vec */
   void (*VecCreateByVec)          (void **des_vec, void *src_vec);
   void (*VecCreateByMat)          (void **vec, void *mat);
   void (*VecDestroy)              (void **vec);
   void (*VecSetRandomValue)       (void  *vec);
   void (*VecAXPBY)                (MONA_DOUBLE  a, void *vec_x, MONA_DOUBLE b, void *vec_y); 
   void (*VecInnerProd)            (MONA_DOUBLE *value, void *vec_x, void *vec_y);
   void (*VecLocalInnerProd)       (MONA_DOUBLE *value, void *vec_x, void *vec_y);
   /* vec_y = mat * vec_x */
   void (*MatDotVec)               (void *vec_y, void *mat, void *vec_x);
   /* --------------------------------------------------------------------------------------------- */

   /* OPTION Mat operations used in Multigird to create matrix */
   /* 
    * row and col are global indices 
    * pre calloc is nnz for each row or nnzs and set zeros 
    *
    * TODO: is it enough to parallel matrix? For example, PETSC Mat
    */
   void (*MatCreate)               (void **mat, MONA_INT nrows, MONA_INT ncols, 
	                            MONA_INT nnz, MONA_INT *nnzs);
   /* 
    * Set mat[row][col[i]] = values[i] for i=0...length-1
    * If row = -1,   mat is dense matrix.
    * If col = NULL, mat[row][0...ncols-1] are nonzeros.
    */
   void (*MatSetValues)            (void  *mat, MONA_INT row, MONA_INT *col, MONA_INT length, MONA_DOUBLE *values);
   void (*MatAssemble)             (void  *mat);
   void (*MatDestroy)              (void **mat);

   /* OPTION */
   /* 
    * temporary workspace and parameters are in xxx_solver_workspace.
    * User can call LinearSolverSetXX(XX xx, MONA_OPS *ops) to set parameter in xxx_solver_workspace.
    */
   void (*LinearSolver)            (void *mat, void **rhs, void **vec, MONA_INT num, 
	                            struct MONA_OPS_ *ops);
   void *linear_solver_workspace;
   /* 
    * INOUT  multivec end
    * multivec[start] ... multivec[*end-1] will be ORTH NORM to multivec[0] ... multivec[start-1]
    * Linearly dependent vectors in multivec[start] ... multivec[*end-1] will been deleted.
    * *end will be return.
    */
   void (*Orthonormalization)      (void **multivec, MONA_INT start, MONA_INT *end, void *spd_mat, 
	                            struct MONA_OPS_ *ops);
   void *orthonormalization_workspace;
   void (*EigenSolver)             (void *A, void *B, MONA_DOUBLE *eigenvalues, void **multivec, MONA_INT nev, 
	                            struct MONA_OPS_ *ops);
   void *eigen_solver_workspace;
   /* 
    * IN   mat[0]       vec[0]
    * OUT  mat[1]   ... mat[num_levels-1]
    *      vec[1]   ... vec[num_levels-1]
    *      interp[0]... interp[num_levels-1]
    *      1->0         num_levels-2->num_levels-1
    * If user does not give this function, MONA should call HYPRE to do this.
    * #define MONA_USE_HYPRE 1
    * */
   void (*MultigridSolver)         (void **mat, void **interp, void **vec, MONA_INT num_levels, 
	                            struct MONA_OPS_ *ops);
   void *multigrid_solver_workspace;

   /* OPTION */
   void (*MultiVecCreateByVec)      (void ***multivec, MONA_INT num_vecs, void *vec, 
                                     struct MONA_OPS_ *ops);
   void (*MultiVecCreateByMat)      (void ***multivec, MONA_INT num_vecs, void *mat,
                                     struct MONA_OPS_ *ops);
   void (*MultiVecDestroy)          (void ***multivec, MONA_INT num_vecs, 
	                             struct MONA_OPS_ *ops);
   void (*MultiVecSetRandomValue)   (void **multivec,  MONA_INT start, MONA_INT end, 
	                             struct MONA_OPS_ *ops);
   void (*MatDotMultiVec)           (void **multivec_y, MONA_INT start_y, MONA_INT end_y, 
	                             void *mat, 
	                             void **multivec_x, MONA_INT start_x, MONA_INT end_x, 
                                     struct MONA_OPS_ *ops);
   void (*MultiVecAXPBY)            (MONA_DOUBLE a, 
	                             void **multivec_x, MONA_INT start_x, MONA_INT end_x, 
				     MONA_DOUBLE b, 
				     void **multivec_y, MONA_INT start_y, MONA_INT end_y,
				     struct MONA_OPS_ *ops);
   /*
    * for j = start_y...end_y-1
    *     multivec_y[j] = \sum_{i=start_x}^{end_x} multivec_x[i] coeffi[ldc*(j-start_y)+i] 
    * endfor
    */
   void (*MultiVecLinearComb)      (void **multivec_y,   MONA_INT start_y, MONA_INT end_y, 
	                            void **multivec_x,   MONA_INT start_x, MONA_INT end_x,
                                    MONA_DOUBLE *coeffi, MONA_INT ldc, 
				    struct MONA_OPS_ *ops);
   /*
    * values = x' y 
    * values[ldv*j+i] = multivec_x[i]*multivec_y[j]
    */
   void (*MultiVecInnerProd)       (MONA_DOUBLE *values,  MONA_INT ldv, char *is_sym, 
	                            void **multivec_x, MONA_INT start_x, MONA_INT end_x,
	                            void **multivec_y, MONA_INT start_y, MONA_INT end_y, 
				    struct MONA_OPS_ *ops);
   void (*MultiVecSwap)            (void **multivec_x, MONA_INT start_x, MONA_INT end_x,
	                            void **multivec_y, MONA_INT start_y, MONA_INT end_y, 
                                    struct MONA_OPS_ *ops);
   void (*MultiVecPrint)           (void **multivec_x, MONA_INT n);
   void (*GetVecFromMultiVec)      (void **vec, void **multivec, MONA_INT j);
   void (*RestoreVecForMultiVec)   (void **vec, void **multivec, MONA_INT j);

}MONA_OPS;

void MONA_OPS_Create (MONA_OPS **ops);
void MONA_OPS_Setup  (MONA_OPS  *ops);
void MONA_OPS_Destroy(MONA_OPS **ops);

void MONA_OPS_SetOrthonormalizationWorkspace(void *orthonormalization_workspace, MONA_OPS *ops);
void MONA_OPS_SetLinearSolverWorkspace(void *linear_solver_workspace, MONA_OPS *ops);
void MONA_OPS_SetEigenSolverWorkspace(void *eigen_solver_workspace, MONA_OPS *ops);

#endif
