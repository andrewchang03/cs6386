#include <iostream>
#include <fstream>

#include <cuda_runtime_api.h>
#include <cusparse.h>

#include <stdio.h>  // printf
#include <stdlib.h> // EXIT_FAILURE

#define CHECK_CUDA(func)                                               \
    {                                                                  \
        cudaError_t status = (func);                                   \
        if (status != cudaSuccess)                                     \
        {                                                              \
            printf("CUDA API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cudaGetErrorString(status), status);      \
            return EXIT_FAILURE;                                       \
        }                                                              \
    }

#define CHECK_CUSPARSE(func)                                               \
    {                                                                      \
        cusparseStatus_t status = (func);                                  \
        if (status != CUSPARSE_STATUS_SUCCESS)                             \
        {                                                                  \
            printf("CUSPARSE API failed at line %d with error: %s (%d)\n", \
                   __LINE__, cusparseGetErrorString(status), status);      \
            return EXIT_FAILURE;                                           \
        }                                                                  \
    }

int *hRowPtrs, *hColInds, *dRowPtrs, *dColInds;
float *hData, *dData;

template <typename T>
void print_from_device(T *dBuffer, size_t count)
{
    T *hBuffer = (T *)malloc(count * sizeof(T));
    cudaMemcpy(hBuffer, dBuffer, count * sizeof(T), cudaMemcpyDeviceToHost);

    for (int i = 0; i < count; ++i)
        std::cout << hBuffer[i] << " ";
    std::cout << std::endl;

    free(hBuffer);
}

void read_data(std::string fName, size_t nrows, size_t nnz)
{
    std::string path = "data/" + fName + "/" + fName + "-";

    // init files
    std::ifstream fRowPtrs(path + "rowptrs", std::ios::binary);
    std::ifstream fColInds(path + "colinds", std::ios::binary);
    std::ifstream fData(path + "data", std::ios::binary);

    // read files into host memory
    fRowPtrs.read(reinterpret_cast<char *>(hRowPtrs), (nrows + 1) * sizeof(int));
    fColInds.read(reinterpret_cast<char *>(hColInds), nnz * sizeof(int));
    fData.read(reinterpret_cast<char *>(hData), nnz * sizeof(float));

    fRowPtrs.close();
    fColInds.close();
    fData.close();

    // for (size_t i = 0; i < nnz; ++i) {
    //     std::cout << hData[i] << " ";
    // }
    // std::cout << std::endl;
}

int main(int argc, char *argv[])
{
    // READ DATA
    char *fName = argv[1];
    int nrows = std::atoi(argv[2]);
    int ncols = std::atoi(argv[3]);
    int nnz = std::atoi(argv[4]);

    hRowPtrs = (int *)calloc(nrows + 1, sizeof(int));
    hColInds = (int *)calloc(nnz, sizeof(int));
    hData = (float *)calloc(nnz, sizeof(float));

    read_data(fName, nrows, nnz);

    CHECK_CUDA(cudaMalloc((void **)&dRowPtrs, (nrows + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dColInds, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dData, nnz * sizeof(float)))

    CHECK_CUDA(cudaMemcpy(dRowPtrs, hRowPtrs, (nrows + 1) * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dColInds, hColInds, nnz * sizeof(int),
                          cudaMemcpyHostToDevice))
    CHECK_CUDA(cudaMemcpy(dData, hData, nnz * sizeof(float),
                          cudaMemcpyHostToDevice))

    // CREATE SPARSE MATRICES U (matA), UT (matB)
    cusparseHandle_t handle = NULL;
    CHECK_CUSPARSE(cusparseCreate(&handle))
    cusparseSpMatDescr_t matA, matB, matC; // matC is the product AB

    CHECK_CUSPARSE(cusparseCreateCsr(&matA, nrows, ncols, nnz,
                                     dRowPtrs, dColInds, dData,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    // TRANSPOSE U (matA)
    int *dCscColOffsets, *dCscRowInds;
    float *dCscData;
    CHECK_CUDA(cudaMalloc(&dCscColOffsets, (ncols + 1) * sizeof(int)))
    CHECK_CUDA(cudaMalloc(&dCscRowInds, nnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc(&dCscData, nnz * sizeof(float)))
    size_t cscBufferSize = 0;
    void *dCscBuffer = nullptr;
    CHECK_CUSPARSE(cusparseCsr2cscEx2_bufferSize(
        handle, nrows, ncols, nnz,
        dData, dRowPtrs, dColInds,
        dCscData, dCscColOffsets, dCscRowInds,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        &cscBufferSize))
    CHECK_CUDA(cudaMalloc(&dCscBuffer, cscBufferSize))
    CHECK_CUSPARSE(cusparseCsr2cscEx2(
        handle, nrows, ncols, nnz,
        dData, dRowPtrs, dColInds,
        dCscData, dCscColOffsets, dCscRowInds,
        CUDA_R_32F, CUSPARSE_ACTION_NUMERIC,
        CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_CSR2CSC_ALG1,
        dCscBuffer))
    CHECK_CUSPARSE(cusparseCreateCsr(&matB, ncols, nrows, nnz,
                                     dCscColOffsets, dCscRowInds, dCscData,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    int *dSpgemmRowPtrs, *dSpgemmColInds;
    float *dSpgemmData;
    CHECK_CUDA(cudaMalloc((void **)&dSpgemmRowPtrs,
                          (nrows + 1) * sizeof(int)))
    CHECK_CUSPARSE(cusparseCreateCsr(&matC, nrows, nrows, 0,
                                     dSpgemmRowPtrs, NULL, NULL,
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F))

    // SPGEMM COMPUTATION AB (U * UT)
    float alpha = 1.0f;
    float beta = 0.0f;
    cusparseOperation_t opA = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseOperation_t opB = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cudaDataType computeType = CUDA_R_32F;
    void *dSpgemmBuffer1 = NULL, *dSpgemmBuffer2 = NULL;
    size_t spgemmBufferSize1 = 0, spgemmBufferSize2 = 0;

    cusparseSpGEMMDescr_t spgemmDesc;
    CHECK_CUSPARSE(cusparseSpGEMM_createDescr(&spgemmDesc))

    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_ALG2,
                                      spgemmDesc, &spgemmBufferSize1, NULL))
    CHECK_CUDA(cudaMalloc((void **)&dSpgemmBuffer1, spgemmBufferSize1))
    CHECK_CUSPARSE(
        cusparseSpGEMM_workEstimation(handle, opA, opB,
                                      &alpha, matA, matB, &beta, matC,
                                      computeType, CUSPARSE_SPGEMM_ALG2,
                                      spgemmDesc, &spgemmBufferSize1, dSpgemmBuffer1))
    CHECK_CUSPARSE(
        cusparseSpGEMM_compute(handle, opA, opB,
                               &alpha, matA, matB, &beta, matC,
                               computeType, CUSPARSE_SPGEMM_ALG2,
                               spgemmDesc, &spgemmBufferSize2, NULL))
    CHECK_CUDA(cudaMalloc((void **)&dSpgemmBuffer2, spgemmBufferSize2))
    CHECK_CUSPARSE(cusparseSpGEMM_compute(handle, opA, opB,
                                          &alpha, matA, matB, &beta, matC,
                                          computeType, CUSPARSE_SPGEMM_ALG2,
                                          spgemmDesc, &spgemmBufferSize2, dSpgemmBuffer2))

    int64_t spgemmNrows, spgemmNcols, spgemmNnz;
    CHECK_CUSPARSE(cusparseSpMatGetSize(matC, &spgemmNrows, &spgemmNcols,
                                        &spgemmNnz))
    CHECK_CUDA(cudaMalloc((void **)&dSpgemmColInds, spgemmNnz * sizeof(int)))
    CHECK_CUDA(cudaMalloc((void **)&dSpgemmData, spgemmNnz * sizeof(float)))
    CHECK_CUSPARSE(
        cusparseCsrSetPointers(matC, dSpgemmRowPtrs, dSpgemmColInds, dSpgemmData))
    CHECK_CUSPARSE(
        cusparseSpGEMM_copy(handle, opA, opB,
                            &alpha, matA, matB, &beta, matC,
                            computeType, CUSPARSE_SPGEMM_ALG2, spgemmDesc))

    std::cout << spgemmNrows << " " << spgemmNcols << std::endl;
    // print_from_device(dSpgemmData, spgemmNnz);

    // DESTROY
    CHECK_CUSPARSE(cusparseSpGEMM_destroyDescr(spgemmDesc))
    CHECK_CUSPARSE(cusparseDestroySpMat(matA))
    CHECK_CUSPARSE(cusparseDestroySpMat(matB))
    CHECK_CUSPARSE(cusparseDestroySpMat(matC))
    CHECK_CUSPARSE(cusparseDestroy(handle))

    CHECK_CUDA(cudaFree(dSpgemmBuffer1))
    CHECK_CUDA(cudaFree(dSpgemmBuffer2))
    CHECK_CUDA(cudaFree(dRowPtrs))
    CHECK_CUDA(cudaFree(dColInds))
    CHECK_CUDA(cudaFree(dData))
    CHECK_CUDA(cudaFree(dCscColOffsets))
    CHECK_CUDA(cudaFree(dCscRowInds))
    CHECK_CUDA(cudaFree(dCscData))
    CHECK_CUDA(cudaFree(dSpgemmRowPtrs))
    CHECK_CUDA(cudaFree(dSpgemmColInds))
    CHECK_CUDA(cudaFree(dSpgemmData))

    free(hRowPtrs);
    free(hColInds);
    free(hData);

    return 0;
}