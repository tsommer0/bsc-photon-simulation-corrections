import math
import warnings
from functools import partial
from typing import Optional, Union

import numpy as np
import ot as pot
import torch


class OTPlanSampler:
    """OTPlanSampler implements sampling coordinates according to an OT plan (wrt squared Euclidean
    cost) with different implementations of the plan calculation."""

    def __init__(
        self,
        method: str,
        reg: float = 0.05,
        reg_m: float = 1.0,
        normalize_cost: bool = False,
        num_threads: Union[int, str] = 1,
        warn: bool = True,
    ) -> None:
        """Initialize the OTPlanSampler class.

        Parameters
        ----------
        method: str
            choose which optimal transport solver you would like to use.
            Currently supported are ["exact", "sinkhorn", "unbalanced",
            "partial"] OT solvers.
        reg: float, optional
            regularization parameter to use for Sinkhorn-based iterative solvers.
        reg_m: float, optional
            regularization weight for unbalanced Sinkhorn-knopp solver.
        normalize_cost: bool, optional
            normalizes the cost matrix so that the maximum cost is 1. Helps
            stabilize Sinkhorn-based solvers. Should not be used in the vast
            majority of cases.
        num_threads: int or str, optional
            number of threads to use for the "exact" OT solver. If "max", uses
            the maximum number of threads.
        warn: bool, optional
            if True, raises a warning if the algorithm does not converge
        """
        # ot_fn should take (a, b, M) as arguments where a, b are marginals and
        # M is a cost matrix
        if method == "exact":
            self.ot_fn = partial(pot.emd, numThreads=num_threads)
        elif method == "sinkhorn":
            self.ot_fn = partial(pot.bregman.sinkhorn_log, reg=reg, numItermax=1000)
        elif method == "unbalanced":
            self.ot_fn = partial(pot.unbalanced.sinkhorn_unbalanced, reg=reg, reg_m=reg_m)
        elif method == "partial":
            self.ot_fn = partial(pot.partial.entropic_partial_wasserstein, reg=reg)
        else:
            raise ValueError(f"Unknown method: {method}")
        self.reg = reg
        self.reg_m = reg_m
        self.normalize_cost = normalize_cost
        self.warn = warn

    def get_map(self, x0, x1):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a = torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=x0.device)
        b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = pot.dist(x0, x1)
        if self.normalize_cost:
            M = M / (M.max().item() + 1e-6)  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M)
        if not torch.all(torch.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if torch.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = torch.ones_like(p) / p.numel()
        return p
    
    def get_weighted_map(self, x0, x1, y0):
        """Compute the OT plan (wrt squared Euclidean cost) between a source and a target
        minibatch.

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch

        Returns
        -------
        p : numpy array, shape (bs, bs)
            represents the OT plan between minibatches
        """
        a = y0 / y0.sum()
        a = a.to(x0.device)
        b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = pot.dist(x0, x1)
        if self.normalize_cost:
            M = M / (M.max().item() + 1e-6)  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M)
        if not torch.all(torch.isfinite(p)):
            print("ERROR: p is not finite")
            print(p)
            print("Cost mean, max", M.mean(), M.max())
            print(x0, x1)
        if torch.abs(p.sum()) < 1e-8:
            if self.warn:
                warnings.warn("Numerical errors in OT plan, reverting to uniform plan.")
            p = torch.ones_like(p) / p.numel()
        return p

    def sample_map(self, pi, batch_size, replace=True):
        r"""Draw source and target samples from pi  $(x,z) \sim \pi$

        Parameters
        ----------
        pi : numpy array, shape (bs, bs)
            represents the source minibatch
        batch_size : int
            represents the OT plan between minibatches
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        (i_s, i_j) : tuple of numpy arrays, shape (bs, bs)
            represents the indices of source and target data samples from $\pi$
        """
        p = pi.flatten()
        p = p / p.sum()
        choices = torch.multinomial(
            p, 
            batch_size, 
            replacement=replace
            )
        return torch.div(choices, pi.shape[1], rounding_mode='floor'), choices % pi.shape[1]

    def sample_plan(self, x0, x1, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the source minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return x0[i], x1[j]

    def sample_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_map(x0, x1)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )
    
    def sample_weighted_plan_with_labels(self, x0, x1, y0=None, y1=None, replace=True):
        r"""Compute the OT plan $\pi$ (wrt squared Euclidean cost) between a source and a target
        minibatch and draw source and target labeled samples from pi $(x,z) \sim \pi$

        Parameters
        ----------
        x0 : Tensor, shape (bs, *dim)
            represents the source minibatch
        x1 : Tensor, shape (bs, *dim)
            represents the target minibatch
        y0 : Tensor, shape (bs)
            represents the source label minibatch
        y1 : Tensor, shape (bs)
            represents the target label minibatch
        replace : bool
            represents sampling or without replacement from the OT plan

        Returns
        -------
        x0[i] : Tensor, shape (bs, *dim)
            represents the source minibatch drawn from $\pi$
        x1[j] : Tensor, shape (bs, *dim)
            represents the target minibatch drawn from $\pi$
        y0[i] : Tensor, shape (bs, *dim)
            represents the source label minibatch drawn from $\pi$
        y1[j] : Tensor, shape (bs, *dim)
            represents the target label minibatch drawn from $\pi$
        """
        pi = self.get_weighted_map(x0, x1, y0)
        i, j = self.sample_map(pi, x0.shape[0], replace=replace)
        return (
            x0[i],
            x1[j],
            y0[i] if y0 is not None else None,
            y1[j] if y1 is not None else None,
        )
    
    '''def sample_sommer(self, x0, x1, y0, y1, tolerance=0.5):
        r"""Custom sampling function for the Sommer project for discarding outliers.
        
        """
        a = torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=x0.device, dtype=torch.float32)
        b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device, dtype=torch.float32)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = pot.dist(x0,x1)
        M = M.to(torch.float32)
        #print(M)
        #print(x0)
        #print(x1)
        #print(M)
        if self.normalize_cost:
            M = M / (M.max().item() + 1e-6)  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M)
        
        #row_sums = p.sum(dim=0)
        #col_sums = p.sum(dim=1)

        rows, cols = torch.where(p > 0.4*p.max())

        return (
            x0[rows],
            x1[cols],
            y0[rows] if y0 is not None else None,
            y1[cols] if y1 is not None else None,
        )'''
        
        
    '''#unmatch_target = row_sums < b * tolerance
        #unmatch_base = col_sums < a * tolerance

        #indices = torch.nonzero(unmatch_target, as_tuple=False).squeeze(1)
        #indices = torch.nonzero(p > 0.0)

        
        indices = torch.argmax(p, axis=1)
        unmatch = row_sums < b*tolerance
        indices = indices[~unmatch]
        return (
            x0[~unmatch],
            x1[indices],
            y0[~unmatch] if y0 is not None else None,
            y1[indices] if y1 is not None else None,
        )'''

        
    def sample_sommer(self, x0, x1, y0, y1, tolerance=0.5):
        r"""Custom sampling function for the Sommer project for discarding outliers.
        
        """
        a = torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=x0.device)
        b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=x1.device)
        if x0.dim() > 2:
            x0 = x0.reshape(x0.shape[0], -1)
        if x1.dim() > 2:
            x1 = x1.reshape(x1.shape[0], -1)
        M = pot.dist(x0,x1)
        if self.normalize_cost:
            M = M / (M.max().item() + 1e-6)  # should not be normalized when using minibatches
        p = self.ot_fn(a, b, M)
        
        #row_sums = p.sum(dim=0)
        #col_sums = p.sum(dim=1)

        rows, cols = torch.where(p > 0.97*p.max())

        x0=x0[rows]
        x1=x1[cols]
        y0=y0[rows] if y0 is not None else None
        y1=y1[cols] if y1 is not None else None

        if y0 is not None and y1 is not None:
            correct = (y0[:, :2] == y1).all(dim=1)
            x0 = x0[correct]
            x1 = x1[correct]
            y0 = y0[correct]
            y1 = y1[correct]

        return (
            x0,
            x1,
            y0 if y0 is not None else None,
            y1 if y1 is not None else None,
        )
        
        

        

    def sample_trajectory(self, X):
        """Compute the OT trajectories between different sample populations moving from the source
        to the target distribution.

        Parameters
        ----------
        X : Tensor, (bs, times, *dim)
            different populations of samples moving from the source to the target distribution.

        Returns
        -------
        to_return : Tensor, (bs, times, *dim)
            represents the OT sampled trajectories over time.
        """
        times = X.shape[1]
        pis = []
        for t in range(times - 1):
            pis.append(self.get_map(X[:, t], X[:, t + 1]))

        indices = [torch.arange(X.shape[0], device=X.device)]
        for pi in pis:
            # Gather the rows for the current indices
            rows = pi[indices[-1]]
            # Normalize each row and sample one index per row
            sampled = torch.multinomial(rows / rows.sum(dim=1, keepdim=True), 1).squeeze(1)
            indices.append(sampled)

        to_return = []
        for t in range(times):
            to_return.append(X[:, t][indices[t]])
        to_return = torch.stack(to_return, dim=1)
        return to_return


def wasserstein(
    x0: torch.Tensor,
    x1: torch.Tensor,
    method: Optional[str] = None,
    reg: float = 0.05,
    power: int = 2,
    **kwargs,
) -> float:
    """Compute the Wasserstein (1 or 2) distance (wrt Euclidean cost) between a source and a target
    distributions.

    Parameters
    ----------
    x0 : Tensor, shape (bs, *dim)
        represents the source minibatch
    x1 : Tensor, shape (bs, *dim)
        represents the source minibatch
    method : str (default : None)
        Use exact Wasserstein or an entropic regularization
    reg : float (default : 0.05)
        Entropic regularization coefficients
    power : int (default : 2)
        power of the Wasserstein distance (1 or 2)
    Returns
    -------
    ret : float
        Wasserstein distance
    """
    assert power == 1 or power == 2
    # ot_fn should take (a, b, M) as arguments where a, b are marginals and
    # M is a cost matrix
    if method == "exact" or method is None:
        ot_fn = pot.emd2
    elif method == "sinkhorn":
        ot_fn = partial(pot.sinkhorn2, reg=reg)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure everything is torch and on the same device, avoid numpy
    device = x0.device
    a = torch.full((x0.shape[0],), 1.0 / x0.shape[0], device=device)
    b = torch.full((x1.shape[0],), 1.0 / x1.shape[0], device=device)
    if x0.dim() > 2:
        x0 = x0.reshape(x0.shape[0], -1)
    if x1.dim() > 2:
        x1 = x1.reshape(x1.shape[0], -1)
    M = torch.cdist(x0, x1)
    if power == 2:
        M = M**2
    # If ot_fn is from POT, it may require numpy arrays, but we want to keep things on device if possible
    # Try to use torch tensors, but if ot_fn is from POT, convert to cpu/numpy
    # Check if ot_fn is from POT (has __module__ 'ot.')
    if hasattr(ot_fn, '__module__') and ot_fn.__module__.startswith("ot."):
        ret = ot_fn(
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            M.detach().cpu().numpy(),
            numItermax=int(1e7)
        )
    else:
        ret = ot_fn(a, b, M)
    if power == 2:
        ret = math.sqrt(ret)
    return ret
