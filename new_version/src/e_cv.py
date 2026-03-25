"""
5_cv.py — The Pipeline's Time Series Cross-Validation Firewall

This module serves as the core temporal integrity firewall of the MFW pipeline.
It implements the Purged K-Fold and Combinatorial Purged K-Fold (CPCV) cross-validation
techniques strictly according to Marcos López de Prado's "Advances in Financial Machine Learning"
(Chapters 7 and 12). 

Its sole responsibility is dropping overlapping observation horizons (via train purging) 
and eliminating serial correlation drift (via test embargoing) across temporal splits.
"""

import itertools
import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold, KFold

logger = logging.getLogger(__name__)


def get_train_times(t1: pd.Series, test_times: pd.Series, is_contiguous: bool = False) -> pd.Series:
    """Purges overlapping observations from the training candidate set.
    
    This function enforces strict causality. It removes training times whose evaluation 
    endpoints overlap with the test fold boundaries, protecting the model from look-ahead leakage.

    Args:
        t1: Full mapping of t0 to t1 observation windows safe for cross-validation.
        test_times: Full mapping of start to end windows tracking the isolated test fold segment.
        is_contiguous: Flag indicating if the test times form a single continuous block.

    Returns:
        pd.Series: A purged subset of t1 explicitly safe to train upon.
    """
    train_t1 = t1.copy(deep=True)
    if test_times.empty:
        return train_t1
        
    # FIX-5: Optimize get_train_times for contiguous test sets
    if is_contiguous:
        t_start = test_times.index.min()
        t_end = test_times.max()
        
        # Condition 1: train_t0 starts inside [t_start, t_end]
        overlap1 = train_t1.index[(train_t1.index >= t_start) & (train_t1.index <= t_end)]
        
        # Condition 2: train_t1 ends inside [t_start, t_end]
        overlap2 = train_t1.index[(train_t1 >= t_start) & (train_t1 <= t_end)]
        
        # Condition 3: test_window envelope is fully enclosed within [train_t0, train_t1]
        overlap3 = train_t1.index[(train_t1.index <= t_start) & (train_t1 >= t_end)]
        
        drop_idx = overlap1.union(overlap2).union(overlap3)
        train_t1 = train_t1.drop(drop_idx, errors='ignore')
        
    else:
        # Fall back to row-by-row loop only for non-contiguous test groups
        for t_start, t_end in test_times.items():
            overlap1 = train_t1.index[(train_t1.index >= t_start) & (train_t1.index <= t_end)]
            overlap2 = train_t1.index[(train_t1 >= t_start) & (train_t1 <= t_end)]
            overlap3 = train_t1.index[(train_t1.index <= t_start) & (train_t1 >= t_end)]
            
            drop_idx = overlap1.union(overlap2).union(overlap3)
            train_t1 = train_t1.drop(drop_idx, errors='ignore')
        
    return train_t1


def get_embargo_times(times: pd.Index, pct_embargo: float = 0.01) -> pd.Series:
    """Computes mapped extension points to implement the exact embargo gap window.
    
    After dropping observations bridging the test set (purging), we embargo observations 
    trailing the test set to eliminate serial correlation completely.
    
    Args:
        times: Monotonic temporal index of the underlying observations.
        pct_embargo: Temporal exclusion density dictating the length of the extension. Default 0.01.

    Returns:
        pd.Series: A forward map extending each discrete input stamp into the embargo boundary.
    """
    embargo_size = int(len(times) * pct_embargo)
    embargo_size = max(1, embargo_size)
    
    embargo_times = pd.Series(index=times, dtype='datetime64[ns]')
    
    for i, t in enumerate(times):
        idx_ext = min(i + embargo_size, len(times) - 1)
        embargo_times.loc[t] = times[idx_ext]
        
    return embargo_times


# FIX-4: Extract shared embargo and purge logic into a helper function
def _apply_embargo_and_purge(
    t1: pd.Series,
    X_index: pd.Index,
    test_indices: np.ndarray,
    embargo_mapping: pd.Series,
    is_contiguous: bool = False,
) -> Tuple[pd.Series, pd.Series]:
    """Applies embargo extension to test times, then purges overlapping training candidates.

    Args:
        t1: Full t0 to t1 observation mapping.
        X_index: The datetime index of the feature matrix X.
        test_indices: Positional indices of the test set in X.
        embargo_mapping: Forward map from each timestamp to its embargo boundary.
        is_contiguous: Flag passed down to get_train_times for processing optimization.

    Returns:
        Tuple of (purged_train_t1, embargoed_test_times).
    """
    test_times = t1.loc[X_index[test_indices]].copy()
    
    for t0_test, t1_test in test_times.items():
        if pd.notna(t1_test):
            # FIX-1: Replace exact membership check with searchsorted lookup safely bridging timelines
            loc = embargo_mapping.index.searchsorted(t1_test, side='right')
            loc = min(loc, len(embargo_mapping) - 1)
            test_times.loc[t0_test] = embargo_mapping.iloc[loc]
            
    train_candidates = t1.loc[X_index].drop(X_index[test_indices])
    train_t1 = get_train_times(train_candidates, test_times, is_contiguous=is_contiguous)
    
    return train_t1, test_times


class PurgedKFold(_BaseKFold):
    """Purged and Embargoed K-Fold Cross Validation.

    Splitting method evaluating static block allocations. It dynamically resolves 
    test envelopes, pads them chronologically with an embargo fraction, and drops 
    conflicting points.
    """
    
    def __init__(self, n_splits=6, t1=None, pct_embargo=0.01):
        if t1 is None:
            raise ValueError("t1 tracking Series MUST be provided.")
        self.t1 = t1
        self.pct_embargo = pct_embargo
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        if not X.index.is_monotonic_increasing:
            raise ValueError("X.index must be strictly monotonic increasing.")
        if not X.index.isin(self.t1.index).all():
            raise ValueError("All chronological X indices must align natively in provided t1 bounds.")
            
        kf = KFold(n_splits=self.n_splits, shuffle=False)
        embargo_mapping = get_embargo_times(X.index, self.pct_embargo)
        
        fold_idx = 1
        for raw_train, test_indices in kf.split(X):
            
            # FIX-4: Call the extracted helper using contiguous optimization
            train_t1, test_times = _apply_embargo_and_purge(
                self.t1, X.index, test_indices, embargo_mapping, is_contiguous=True
            )
            
            if len(train_t1) == 0:
                logger.warning("PurgedKFold Fold %d: All training features dropped under purging constraints.", fold_idx)
            
            logger.info("PurgedKFold Fold %d | Train: %d elements | Test: %d elements", fold_idx, len(train_t1), len(test_indices))
            
            train_indices_array = X.index.get_indexer(train_t1.index)
            yield train_indices_array, test_indices
            fold_idx += 1

    def __repr__(self):
        return f"PurgedKFold(n_splits={self.n_splits}, pct_embargo={self.pct_embargo})"


# FIX-3: Inherit from _BaseKFold
class CombinatorialPurgedKFold(_BaseKFold):
    """Combinatorial Purged Cross-Validation (CPCV).

    Generates comprehensive combinations of evaluation folds yielding 
    completely orthogonal out-of-sample backtest paths. 
    """
    
    def __init__(self, n_splits=6, n_test_splits=2, t1=None, pct_embargo=0.01):
        """Initializes CPCV structural allocations.
        
        Args:
            n_splits (int): The total number of generic partitions maping the dataset.
            n_test_splits (int): The number of specific groups extracted to form testing bounds.
            t1 (pd.Series): Target sequence limits supporting evaluation lengths.
            pct_embargo (float): Standard time mapping offset isolating correlation drift.
        """
        # FIX-3: Call super init for sklearn compatibility structurally
        super().__init__(n_splits=n_splits, shuffle=False, random_state=None)
        
        if t1 is None:
            raise ValueError("The parameter `t1` MUST be provided.")
        # FIX-2: Fix CPCV validation guard (off-by-one) allowing exact splits.
        if n_test_splits > n_splits / 2:
            raise ValueError("n_test_splits (k) must be strictly less than or equal to n_splits/2 (N/2).")
            
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        self.phi = math.comb(n_splits, n_test_splits)
        phi_paths = int(self.phi * self.n_test_splits / self.n_splits)
        
        logger.info(
            "Initialized CPCV: N=%d, k=%d. "
            "Total combinations: C(%d, %d) = %d. "
            "Total structural phi paths C(N, k) * k / N = %d",
            n_splits, n_test_splits, n_splits, n_test_splits, self.phi, phi_paths
        )

    # FIX-3: Implement get_n_splits sklearn standard 
    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        """Returns the total number of combinations (self.phi)."""
        return self.phi

    def split(self, X, y=None, groups=None):
        if not X.index.is_monotonic_increasing:
            raise ValueError("X.index must execute with chronological strictly monotonic bounds.")
        if not X.index.isin(self.t1.index).all():
            raise ValueError("All chronological X indices must align correctly onto provided t1 mapping ranges.")
            
        total_len = len(X)
        group_size = total_len // self.n_splits
        groups_list = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else total_len
            groups_list.append(np.arange(start, end))
            
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        embargo_mapping = get_embargo_times(X.index, self.pct_embargo)
        
        for combo_idx, test_group_indices in enumerate(combinations):
            test_indices = np.concatenate([groups_list[i] for i in test_group_indices])
            
            # Non-contiguous indicator resolution accurately assigning limits
            contiguous = True
            if len(test_group_indices) > 1:
                diffs = np.diff(test_group_indices)
                if not np.all(diffs == 1):
                    contiguous = False

            # FIX-4: Extract and deploy modularized embargo purge array mapping logic cleanly
            train_t1, test_times = _apply_embargo_and_purge(
                self.t1, X.index, test_indices, embargo_mapping, is_contiguous=contiguous
            )
            
            if len(train_t1) == 0:
                logger.warning(
                    "CPCV Combination ID:%d isolated all possible testing sets under bounds. "
                    "Pushing explicitly empty zero array.", combo_idx
                )
                yield np.array([], dtype=int), test_indices
            else:
                train_indices_array = X.index.get_indexer(train_t1.index)
                yield train_indices_array, test_indices

    def get_backtest_paths(self, X) -> List[List[Tuple[int, int]]]:
        """Maps specific evaluation routes translating CPCV split models onto test indices.
        
        Yields discrete array-sets isolating temporal evaluations for exactly phi logical routes
        dictating completely exhaustive timeline mappings.

        Returns:
            List[List[Tuple[int, int]]]: Logical collection mapping lists of length N.
                Each tuple expresses: (test_group_index, combination_index).
        """
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        
        inventory = {g: [] for g in range(self.n_splits)}
        for c_idx, combo in enumerate(combinations):
            for g in combo:
                inventory[g].append(c_idx)
                
        num_paths = math.comb(self.n_splits - 1, self.n_test_splits - 1)
        
        paths = []
        for _ in range(num_paths):
            path = []
            for g in range(self.n_splits):
                c_idx = inventory[g].pop(0)
                path.append((g, c_idx))
            paths.append(path)
            
        # FIX-6: Add post-hoc validation bounding strict limits evaluating targets thoroughly
        for p_idx, path in enumerate(paths):
            if len(path) != self.n_splits:
                raise RuntimeError(f"Path {p_idx} has {len(path)} tuples, expected exactly {self.n_splits}.")
            groups_in_path = [g for g, c in path]
            if sorted(groups_in_path) != list(range(self.n_splits)):
                raise RuntimeError(f"Path {p_idx} does not contain exactly one of every group index.")
                
        for g, inv in inventory.items():
            if len(inv) > 0:
                raise RuntimeError(f"Inventory for group {g} is not empty after building all paths. Remaining: {inv}")
                
        return paths

    def __repr__(self):
        return f"CombinatorialPurgedKFold(n_splits={self.n_splits}, n_test_splits={self.n_test_splits}, pct_embargo={self.pct_embargo})"