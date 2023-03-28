sparse_status_t mkl_sparse_s_mm 
(const sparse_operation_t operation, //non-tran
 const float alpha, //1.0
 const sparse_matrix_t A, //A
 const struct matrix_descr descr, //general
 const sparse_layout_t layout, //row-wise
 const float *B, //X
 const MKL_INT columns, 
 const MKL_INT ldb, 
 const float beta, 
 float *C, //O
 const MKL_INT ldc
 );


 void SpMMSumCsrNaive(
    const BcastOff& bcast,  //(???)
    const CSRMatrix& csr, //A
    const DType* X, //node feat
    const DType* W, //edge feat (???)
    DType* O 
    );