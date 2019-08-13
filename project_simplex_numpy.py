import numpy as np


def project_simplex(v, z=1.0, axis=-1):
    """
    Implements the algorithm in Figure 1 of
    John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandra,
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions", ICML 2008.
    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    This algorithm project vectors v onto the simplex w >= 0, \sum w_i = z.

    :param v: A numpy array, will be interpreted as a collection of vectors.
    :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
    :param axis: Indicates the axis of v, which defines the vectors to be projected.
    :return: w: result of the projection
    """

    def _project_simplex_2d(v, z):
        """
        Helper function, assuming that all vectors are arranged in rows of v.

        :param v: NxD numpy array; Duchi et al. algorithm is applied to each row in vecotrized form
        :param z: Vectors will be projected onto the z-Simplex: \sum w_i = z.
        :return: w: result of the projection
        """
        shape = v.shape
        if shape[1] == 1:
            w = np.array(v)
            w[:] = z
            return w

        mu = np.sort(v, axis=1)
        mu = np.flip(mu, axis=1)
        cum_sum = np.cumsum(mu, axis=1)
        j = np.expand_dims(np.arange(1, shape[1] + 1), 0)
        rho = np.sum(mu * j - cum_sum + z > 0.0, axis=1, keepdims=True) - 1
        max_nn = cum_sum[np.arange(shape[0]), rho[:, 0]]
        theta = (np.expand_dims(max_nn, -1) - z) / (rho + 1)
        w = (v - theta).clip(min=0.0)
        return w

    shape = v.shape

    if len(shape) == 0:
        return np.array(1.0, dtype=v.dtype)
    elif len(shape) == 1:
        return _project_simplex_2d(np.expand_dims(v, 0), z)[0, :]
    else:
        axis = axis % len(shape)
        t_shape = tuple(range(axis)) + tuple(range(axis + 1, len(shape))) + (axis,)
        tt_shape = tuple(range(axis)) + (len(shape) - 1,) + tuple(range(axis, len(shape) - 1))
        v_t = np.transpose(v, t_shape)
        v_t_shape = v_t.shape
        v_t_unroll = np.reshape(v_t, (-1, v_t_shape[-1]))

        w_t = _project_simplex_2d(v_t_unroll, z)

        w_t_reroll = np.reshape(w_t, v_t_shape)
        return np.transpose(w_t_reroll, tt_shape)


if __name__ == "__main__":

    # violations will be larger for float32
    # precision = np.float32
    precision = np.float64

    z = 1.0
    overall_nonneg_violation = 0.0
    overall_l1_violation = 0.0
    overall_suboptimality = 0.0

    print("")
    print("**********************")
    print("TEST l1-PROJECT VECTOR")
    print("**********************")

    for d in [1, 2, 3, 5]:
        print("")
        print("Dimensions {}".format(d))

        nonneg_violation = 0.0
        l1_violation = 0.0
        suboptimality = 0.0

        for rep in range(100):
            x = np.random.randn(d).astype(precision)
            x_projected = project_simplex(x, z)

            if rep == 0:
                print("x:")
                print(x)
                print("projected:")
                print(x_projected)

            nonneg_violation_cur = -np.min(x_projected.clip(max=0.0))
            l1_violation_cur = np.abs(np.sum(x_projected) - z)

            D = np.sum((x_projected - x)**2)**0.5

            x_perturbed = np.expand_dims(x_projected, -1) + 0.01 * np.random.randn(d, 10000).astype(precision)
            x_perturbed = x_perturbed.clip(min=0.0)
            x_perturbed /= np.sum(x_perturbed, 0, keepdims=True)
            x_perturbed *= z

            D_perturbed = np.sum((np.expand_dims(x, -1) - x_perturbed)**2, 0)**0.5
            suboptimality_cur = -np.min((D_perturbed - D).clip(max=0.0))

            nonneg_violation = max(nonneg_violation_cur, nonneg_violation)
            l1_violation = max(l1_violation_cur, l1_violation)
            suboptimality = max(suboptimality_cur, suboptimality)
            overall_nonneg_violation = max(nonneg_violation_cur, overall_nonneg_violation)
            overall_l1_violation = max(l1_violation_cur, overall_l1_violation)
            overall_suboptimality = max(suboptimality_cur, overall_suboptimality)


        print("Nonnegativety violation {:0.12f}, l1 violation {:0.12f}, suboptimality {:0.12f}".format(
            nonneg_violation, l1_violation, suboptimality))


    print("")
    print("")
    print("")

    print("**********************")
    print("TEST l1-PROJECT TENSOR")
    print("**********************")
    for m in [2, 3, 4]:
        for a in range(m):
            for d in [1, 2, 3, 5]:
                print("")
                print("Modes {}, Axis {}, Dimensions {}".format(m, a, d))

                nonneg_violation = 0.0
                l1_violation = 0.0
                suboptimality = 0.0

                x_shape = (10,) * a + (d,) + (10,) * (m-a-1)
                x = np.random.randn(*x_shape).astype(precision)

                x_projected = project_simplex(x, z, axis=a)

                nonneg_violation_cur = -np.min(x_projected.clip(max=0.0))
                l1_violation_cur = np.max(np.abs(np.sum(x_projected, axis=a) - z))

                D = np.sum((x_projected - x)**2, axis=a)**0.5

                x_perturbed = x_projected + 0.01 * np.random.randn(*x_projected.shape).astype(precision)
                x_perturbed = x_perturbed.clip(min=0.0)
                x_perturbed /= np.sum(x_perturbed, axis=a, keepdims=True)
                x_perturbed *= z

                D_perturbed = np.sum((x_perturbed - x)**2, axis=a)**0.5
                suboptimality_cur = -np.min((D_perturbed - D).clip(max=0.0))

                nonneg_violation = max(nonneg_violation_cur, nonneg_violation)
                l1_violation = max(l1_violation_cur, l1_violation)
                suboptimality = max(suboptimality_cur, suboptimality)
                overall_nonneg_violation = max(nonneg_violation_cur, overall_nonneg_violation)
                overall_l1_violation = max(l1_violation_cur, overall_l1_violation)
                overall_suboptimality = max(suboptimality_cur, overall_suboptimality)

                print("Nonnegativety violation {:0.12f}, l1 violation {:0.12f}, suboptimality {:0.12f}".format(
                    nonneg_violation, l1_violation, suboptimality))

    print("")
    print("")
    print("")
    print("Overall summary:")
    print("")
    print("Nonnegativety violation {:0.12f} detected.".format(overall_nonneg_violation))
    print("l1 violation violation {:0.12f} detected.".format(overall_l1_violation))
    print("suboptimality {:0.12f} detected.".format(overall_suboptimality))
    print("")
    print("All these values should be close to zero.")
    print("")
