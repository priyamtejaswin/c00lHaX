#created at: Sun Sep  8 19:44:28 2019
#created by: Priyam Tejaswin (tejaswin.p@flipkart.com)
#
#Viterbi in Cython.
#Why? Bcuz
#Aag lagadey baby fire.

cimport cython
from cpython cimport array

import numpy as np
cimport numpy as np

INT_DTYPE = np.int  # INT all the way -- weights, counters, etc.
FLOAT_DTYPE = np.float
ctypedef np.int_t INT_DTYPE_t
ctypedef np.float_t FLOAT_DTYPE_t

def reply(int[:] nums):
    cdef double[:] one_arr = np.zeros(len(nums))
    one_arr[0] = 5.5
    print one_arr
    print nums
    return nums[0], one_arr[0]


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)
def decode(int[:] words_ixs, int num_tags, np.ndarray[INT_DTYPE_t, ndim=2] weights):
    # `word_ixs` is the indices of the word, already mapped.
    # `-1` means OOV.
    cdef int size = len(words_ixs)

    cdef np.ndarray[FLOAT_DTYPE_t, ndim=2] delta = np.zeros((size, num_tags),
                                                            dtype=FLOAT_DTYPE)
    cdef np.ndarray[INT_DTYPE_t, ndim=2] psi = np.zeros((size, num_tags),
                                                        dtype=INT_DTYPE)

    list_num_tags = range(num_tags)

    # Stage1
    cdef int t, wix, tix, curr, prev
    cdef np.ndarray[FLOAT_DTYPE_t, ndim=1] temp = np.zeros(num_tags,
                                                           dtype=FLOAT_DTYPE)

    for t in range(size):  # `t` is time.
        wix = words_ixs[t]

        if t == 0:
            # Prev tag is fixed. Current tag (i.e. state)
            # needs to be determined, based on the current
            # word only (prev tag will be None).
            if wix == -1:
                continue

            for tix in list_num_tags:
                delta[t, tix] = weights[wix, tix]

            #_d = [weights[wix, tix] for tix in list_num_tags]
            #_p = np.zeros(num_tags, dtype=int).tolist()

        else:
            # Here, we run the |s|^2 loop.
            # Again, we use the `word` feature only if
            # it is present.

            #_d, _p = [], []
            for curr in list_num_tags:  # `curr` represents the current tag.
                for prev in list_num_tags:
                    temp[prev] = delta[t-1, prev] + weights[prev, curr]
                    if wix != -1:
                        temp[prev] += weights[wix, curr]

                delta[t, curr] = np.max(temp)
                psi[t, curr] = np.argmax(temp)

                #_d.append(np.max(temp))
                #_p.append(np.argmax(temp))

    # Stage2
    cdef np.ndarray[INT_DTYPE_t, ndim=1] decoding = np.zeros(size, dtype=INT_DTYPE)
    cdef int qt = np.argmax(delta[size-1])
    decoding[size-1] = qt

    for t in range(size-1, 0, -1):
        qt = psi[t, qt]
        decoding[t-1] = qt

    return decoding
