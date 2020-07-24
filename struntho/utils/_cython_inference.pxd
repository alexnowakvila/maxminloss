from cython cimport floating

cpdef softmax1_c(floating[::1], floating[::1])

cpdef linear_comb_c(floating, floating, floating[::1], floating[::1])

cpdef floating max_c(floating[::1])

cpdef floating min_c(floating[::1])

cpdef floating logsumexp(floating[:], floating)

cpdef softmax2_c(floating[:, :], floating[:, :])

cpdef apply_log2(floating[:, :], int, int)
            
cpdef apply_log3(floating[:, :, :], int, int, int)
                
cpdef apply_exp2(floating[:, :], int, int)
            
cpdef apply_exp3(floating[:, :, :], int, int, int)

cpdef augment_nodes(floating[:, :],
                   floating[:, :],
                   floating[:, :],
                   floating[:, :],
                   floating[:, :],
                   floating,
                   int,
                   int)
    
cpdef augment_edges(floating[:, :, :],
                    floating[:, :],
                    floating[:, :, :],
                    floating,
                    int,
                    int)

cpdef linear_comb2(int, int, floating, floating,
                floating[:, :], floating[:, :], int)

cpdef linear_comb3(int, int, int, floating alpha, floating,
                floating[:, :, :], floating[:, :, :], int)