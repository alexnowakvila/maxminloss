from cython cimport floating


cpdef enum BLAS_Order:
    RowMajor  # C contiguous
    ColMajor  # Fortran contiguous


cpdef enum BLAS_Trans:
    NoTrans = 110  # correspond to 'n'
    Trans = 116    # correspond to 't'


# BLAS Level 1 ################################################################
cdef floating _dot(int, floating*, int, floating*, int) nogil
cpdef _dot_memview(floating[::1], floating[::1])

cdef floating _asum(int, floating*, int) nogil
cpdef _asum_memview(floating[::1])

cdef void _axpy(int, floating, floating*, int, floating*, int) nogil
cpdef _axpy_memview(floating, floating[::1], floating[::1])

cdef floating _nrm2(int, floating*, int) nogil
cpdef _nrm2_memview(floating[::1])

cdef void _copy(int, floating*, int, floating*, int) nogil
cpdef _copy_memview(floating[::1], floating[::1])

cdef void _scal(int, floating, floating*, int) nogil
cpdef _scal_memview(floating , floating[::1])

cdef void _rotg(floating*, floating*, floating*, floating*) nogil
cpdef _rotg_memview(floating, floating, floating, floating)

cdef void _rot(int, floating*, int, floating*, int, floating, floating) nogil
cpdef _rot_memview(floating[::1], floating[::1], floating, floating)

# BLAS Level 2 ################################################################
cdef void _gemv(int, int, int, int, floating, floating*, int,
                floating*, int, floating, floating*, int) nogil
cpdef _gemv_memview(int, floating, floating[:, :],
                    floating[::1], floating, floating[::1])

cdef void _ger(BLAS_Order, int, int, floating, floating*, int, floating*, int,
               floating*, int) nogil
cpdef _ger_memview(floating, floating[::1], floating[::], floating[:, :])

# BLASLevel 3 ################################################################
cdef void _gemm(int, int, int, int, int, int, floating,
                floating*, int, floating*, int, floating, floating*,
                int) nogil
cpdef _gemm_memview(int, int, floating,
                    floating[:, :], floating[:, :], floating, floating[:, :])