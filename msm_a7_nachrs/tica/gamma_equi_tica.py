"""
@author: yuxuanzhuang
"""

from collections import namedtuple
from numbers import Integral
from typing import Optional, Union, Callable

import numpy as np

from deeptime.decomposition import VAMP, TICA, CovarianceKoopmanModel, TransferOperatorModel
from deeptime.base import EstimatorTransformer
from deeptime.basis import Identity, Observable, Concatenation
from deeptime.covariance import Covariance, CovarianceModel
from deeptime.numeric import spd_inv_split, eig_corr
from deeptime.util.types import to_dataset


class GAMMA_EQUI_VAMP(VAMP):
    r"""Variational approach for Markov processes (VAMP) with a symmetric observable transform.
    """

    def __init__(self, symmetry_fold: int =1,
                 lagtime: Optional[int] = None,
                 dim: Optional[int] = None,
                 var_cutoff: Optional[float] = None,
                 scaling: Optional[str] = None,
                 epsilon: float = 1e-6,
                 observable_transform: Callable[[np.ndarray], np.ndarray] = Identity()):
        super(VAMP, self).__init__()
        self.symmetry_fold = symmetry_fold
        self.dim = dim
        self.var_cutoff = var_cutoff
        self.scaling = scaling
        self.epsilon = epsilon
        self.lagtime = lagtime
        self.observable_transform = observable_transform
        self._covariance_estimator = None  # internal covariance estimator

    _DiagonalizationResults = namedtuple("DiagonalizationResults", ['rank0', 'rankt', 'singular_values',
                                                                    'left_singular_vecs', 'right_singular_vecs'])

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff, symmetry_fold) -> _DiagonalizationResults:
        """Performs SVD on covariance matrices and save left, right singular vectors and values in the model."""
        if covariances.cov_00.shape[0] % symmetry_fold != 0:
            raise ValueError(f"Number of features {covariances.cov_00.shape[0]} must" +
                             f"be divisible by symmetry_fold {symmetry_fold}.")

        subset_rank = covariances.cov_00.shape[0] // symmetry_fold
        cov_00 = covariances.cov_00
        cov_0t = covariances.cov_0t
        cov_tt = covariances.cov_tt

        L0 = spd_inv_split(cov_00, epsilon=epsilon)
        rank0 = L0.shape[1] if L0.ndim == 2 else 1
        Lt = spd_inv_split(cov_tt, epsilon=epsilon)
        rankt = Lt.shape[1] if Lt.ndim == 2 else 1

        W = np.dot(L0.T, cov_0t).dot(Lt)
        from scipy.linalg import svd
        A, s, BT = svd(W, compute_uv=True, lapack_driver='gesvd')

        singular_values = s

        m = CovarianceKoopmanModel.effective_output_dimension(rank0, rankt, dim, var_cutoff, singular_values)

        U = np.dot(L0, A[:, :m])
        V = np.dot(Lt, BT[:m, :].T)

        # scale vectors
        if scaling is not None and scaling in ("km", "kinetic_map"):
            U *= s[np.newaxis, 0:m]  # scaled left singular functions induce a kinetic map
            V *= s[np.newaxis, 0:m]  # scaled right singular functions induce a kinetic map wrt. backward propagator

        return GAMMA_EQUI_VAMP._DiagonalizationResults(
            rank0=rank0, rankt=rankt, singular_values=singular_values, left_singular_vecs=U, right_singular_vecs=V
        )

    def _decompose(self, covariances: CovarianceModel):

        decomposition = self._decomposition(covariances, self.epsilon, self.scaling, self.dim, self.var_cutoff,
                                            self.symmetry_fold)
        return GAMMA_EQUI_CovarianceKoopmanModel(
            self.symmetry_fold,
            decomposition.left_singular_vecs, decomposition.singular_values, decomposition.right_singular_vecs,
            rank_0=decomposition.rank0, rank_t=decomposition.rankt, dim=self.dim,
            var_cutoff=self.var_cutoff, cov=covariances, scaling=self.scaling, epsilon=self.epsilon,
            instantaneous_obs=self.observable_transform,
            timelagged_obs=self.observable_transform
        )

    def fetch_model(self) -> 'GAMMA_EQUI_CovarianceKoopmanModel':
        r""" Finalizes current model and yields new :class:`SymCovarianceKoopmanModel`.
        Returns
        -------
        model : GAMMA_EQUI_CovarianceKoopmanModel
            The estimated model.
        """
        if self._covariance_estimator is not None:
            # This can only occur when partial_fit was called.
            # A call to fit, fit_from_timeseries, fit_from_covariances ultimately always leads to a call to
            # fit_from_covariances which sets the self._covariance_estimator to None.
            self._model = self._decompose(self._covariance_estimator.fetch_model())
            self._covariance_estimator = None
        return self._model
    
    def transform_subunit(self, data, propagate=False):
        r""" Projects given timeseries onto dominant singular functions. This method dispatches to
        :meth:`CovarianceKoopmanModel.transform_subunit`.

        Parameters
        ----------
        data : (T, n) ndarray
            Input timeseries data.
        propagate : bool, default=False
            Whether to apply the Koopman operator after data was transformed into the whitened feature space.

        Returns
        -------
        Y : (T, m) ndarray
            The projected data.
            If `right` is True, projection will be on the right singular functions. Otherwise, projection will be on
            the left singular functions.
        """
        return self.fetch_model().transform_subunit(data, propagate=propagate)


class GAMMA_EQUI_TICA(TICA, GAMMA_EQUI_VAMP):
    def __init__(self, symmetry_fold,
                 lagtime: Optional[int] = None, epsilon: float = 1e-6, dim: Optional[int] = None,
                 var_cutoff: Optional[float] = None, scaling: Optional[str] = 'kinetic_map',
                 observable_transform: Callable[[np.ndarray], np.ndarray] = Identity()):
        GAMMA_EQUI_VAMP.__init__(self, symmetry_fold=symmetry_fold,
                                   lagtime=lagtime, dim=dim, var_cutoff=var_cutoff,
                                   scaling=scaling, epsilon=epsilon,
                                   observable_transform=observable_transform)

    @staticmethod
    def _decomposition(covariances, epsilon, scaling, dim, var_cutoff, symmetry_fold) -> VAMP._DiagonalizationResults:
        print("symmetry_fold", symmetry_fold)
        if covariances.cov_00.shape[0] % symmetry_fold != 0:
            raise ValueError(f"Number of features {covariances.cov_00.shape[0]} must" +
                             f"be divisible by symmetry_fold {symmetry_fold}.")

        cov_00 = covariances.cov_00
        cov_0t = covariances.cov_0t
        cov_tt = covariances.cov_tt
        
        
        from deeptime.numeric import ZeroRankError

        # diagonalize with low rank approximation
        try:
            eigenvalues, eigenvectors, rank = eig_corr(cov_00, cov_0t, epsilon,
                                                       canonical_signs=True, return_rank=True)
        except ZeroRankError:
            raise ZeroRankError('All input features are constant in all time steps. '
                                'No dimension would be left after dimension reduction.')
        if scaling in ('km', 'kinetic_map'):  # scale by eigenvalues
            eigenvectors *= eigenvalues[None, :]
        elif scaling == 'commute_map':  # scale by (regularized) timescales
            lagtime = covariances.lagtime
            timescales = 1. - lagtime / np.log(np.abs(eigenvalues))
            # dampen timescales smaller than the lag time, as in section 2.5 of ref. [5]
            regularized_timescales = 0.5 * timescales * np.maximum(
                np.tanh(np.pi * ((timescales - lagtime) / lagtime) + 1), 0)

            eigenvectors *= np.sqrt(regularized_timescales / 2)

        return GAMMA_EQUI_VAMP._DiagonalizationResults(
            rank0=rank, rankt=rank, singular_values=eigenvalues,
            left_singular_vecs=eigenvectors, right_singular_vecs=eigenvectors
        )


class GAMMA_EQUI_WhiteningTransform(Observable):
    r""" Transformation of symmetric data into a whitened space.
    It is assumed that for a covariance matrix :math:`C` the
    square-root inverse :math:`C^{-1/2}` was already computed. Optionally a mean :math:`\mu` can be provided.
    This yields the transformation
    .. math::
        y = C^{-1/2}(x-\mu).
    Parameters
    ----------
    sqrt_inv_cov : (n, k) ndarray
        Square-root inverse of covariance matrix.
    mean : (n, ) ndarray, optional, default=None
        The mean if it should be subtracted.
    dim : int, optional, default=None
        Additional restriction in the dimension, removes all but the first `dim` components of the output.
    See Also
    --------
    deeptime.numeric.spd_inv_sqrt : Method to obtain (regularized) inverses of covariance matrices.
    """

    def __init__(self, sqrt_inv_cov: np.ndarray, mean: Optional[np.ndarray] = None, dim: Optional[int] = None):
        self.sqrt_inv_cov = sqrt_inv_cov
        self.mean = mean
        self.dim = dim

    def _evaluate(self, x: np.ndarray, nosum=False):
        if self.mean is not None:
            x = x - self.mean
        if nosum:
#            output = np.empty((x.shape[0], x.shape[1], self.dim))
#            for i in range(x.shape[0]):
#                output[i] = x[i].reshape(-1,1) * self.sqrt_inv_cov[..., :self.dim]
#            return output
#            return x.reshape(x.shape[0], x.shape[1], 1) * self.sqrt_inv_cov[..., :self.dim]
            return np.einsum('ij,jk->ijk', x, self.sqrt_inv_cov[:,:self.dim])

        return x @ self.sqrt_inv_cov[..., :self.dim]
        


class GAMMA_EQUI_CovarianceKoopmanModel(CovarianceKoopmanModel, TransferOperatorModel):
    """
    Symmetric transformation version of Covariance Koopman Model
    """
    def __init__(self, symmetrty_fold, instantaneous_coefficients, singular_values, timelagged_coefficients, cov,
                rank_0: int, rank_t: int, dim=None, var_cutoff=None, scaling=None, epsilon=1e-10,
                instantaneous_obs: Callable[[np.ndarray], np.ndarray] = Identity(),
                timelagged_obs: Callable[[np.ndarray], np.ndarray] = Identity()):

        self.symmetry_fold = symmetrty_fold
        self._whitening_instantaneous = GAMMA_EQUI_WhiteningTransform(instantaneous_coefficients,
                                                            cov.mean_0 if cov.data_mean_removed else None)
        self._whitening_timelagged = GAMMA_EQUI_WhiteningTransform(timelagged_coefficients,
                                                        cov.mean_t if cov.data_mean_removed else None)
        TransferOperatorModel.__init__(self, np.diag(singular_values),
                            Concatenation(self._whitening_instantaneous, instantaneous_obs),
                            Concatenation(self._whitening_timelagged, timelagged_obs))
        self._instantaneous_coefficients = instantaneous_coefficients
        self._timelagged_coefficients = timelagged_coefficients
        self._singular_values = singular_values
        self._cov = cov

        self._scaling = scaling
        self._epsilon = epsilon
        self._rank_0 = rank_0
        self._rank_t = rank_t
        self._dim = dim
        self._var_cutoff = var_cutoff
        self._update_output_dimension()
    
    def transform(self, data: np.ndarray, **kw):
#        data = data.reshape(data.shape[0], self.symmetry_fold, -1)
        return self.instantaneous_obs(data)
    
    def transform_subunit(self, data: np.ndarray, **kw):
        transformed = self._whitening_instantaneous._evaluate(data, nosum=True)
        n_feat = transformed.shape[1]
        n_feat_subunit = n_feat // self.symmetry_fold
        transformed = np.sum(transformed.reshape(transformed.shape[0], self.symmetry_fold, n_feat_subunit, -1), axis=2)
        return transformed.transpose(0, 2, 1)