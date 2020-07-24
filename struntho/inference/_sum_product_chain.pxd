from cython cimport floating

cpdef sum_product_c(floating[:, :], floating[:, :, :],
                    floating[:, :], floating[:, :, :])

cpdef viterbi(floating[:, :], floating[:, :], int[::1])