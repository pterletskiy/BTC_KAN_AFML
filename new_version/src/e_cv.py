"""
5_cv.py — The Pipeline's Time-Series Cross-Validation Firewall

This module serves as the core temporal integrity firewall of the MFW pipeline.
It implements the Purged K-Fold and Combinatorial Purged K-Fold (CPCV) cross-validation
techniques strictly according to Marcos López de Prado's "Advances in Financial Machine Learning"
(Chapters 7 and 12). 

Its sole responsibility is dropping or explicitly filtering overlapping observation horizons 
(via train purging) and serial-correlation drift (via test embargoing) spanning temporal splits.
"""

import itertools
import logging
import math
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection._split import _BaseKFold, KFold

logger = logging.getLogger(__name__)


def get_train_times(t1: pd.Series, test_times: pd.Series) -> pd.Series:
    """Purges overlapping observations from the training candidate set.
    
    MLDP Ch 7.4.1. This function enforces strict causality and removes train times
    evaluating endpoints that overlap with the test fold boundaries protecting from look-ahead leakage.

    Args:
        t1: Full mapping of `[t0 -> t1]` observation windows safe for cross-validation evaluation.
        test_times: Full mapping of `[t_start -> t_end]` windows tracking the isolated test fold segment.

    Returns:
        pd.Series: A purged subset of `t1` explicitly safe to train upon.
    """
    train_t1 = t1.copy(deep=True)
    
    for t_start, t_end in test_times.items():
        # Condition 1: train_t0 (index) starts inside [t_start, t_end]
        overlap1 = train_t1.index[
            (train_t1.index >= t_start) & (train_t1.index <= t_end)
        ]
        
        # Condition 2: train_t1 (value) ends inside [t_start, t_end]
        overlap2 = train_t1.index[
            (train_t1 >= t_start) & (train_t1 <= t_end)
        ]
        
        # Condition 3: test_window envelope is fully enclosed within [train_t0, train_t1]
        overlap3 = train_t1.index[
            (train_t1.index <= t_start) & (train_t1 >= t_end)
        ]
        
        drop_idx = overlap1.union(overlap2).union(overlap3)
        train_t1 = train_t1.drop(drop_idx, errors='ignore')
        
    return train_t1


def get_embargo_times(times: pd.Index, pct_embargo: float = 0.01) -> pd.Series:
    """Computes mapped extension points implementing the exact embargo gap window.
    
    MLDP Ch 7.4.2. After dropping observations bridging the test set (purging),
    we embargo observations trailing the test set to eliminate serial correlation.
    
    Args:
        times: Monotonic temporal tracking underlying observations evaluated chronologically.
        pct_embargo: Temporal exclusion density (float dictating length of extension). Default 0.01.

    Returns:
        pd.Series: A forward map extending each discrete input stamp securely into the embargo boundary.
    """
    embargo_size = int(len(times) * pct_embargo)
    embargo_size = max(1, embargo_size)
    
    embargo_times = pd.Series(index=times, dtype='datetime64[ns]')
    
    for i, t in enumerate(times):
        # Extend exactly `embargo_size` bars out, capturing times[-1] if index goes OOB
        idx_ext = min(i + embargo_size, len(times) - 1)
        embargo_times.loc[t] = times[idx_ext]
        
    return embargo_times


class PurgedKFold(_BaseKFold):
    """Purged and Embargoed K-Fold Cross Validation.

    MLDP Ch 7.4.3. Splitting method evaluating static block allocations, dynamically resolving
    test envelopes, padding them chronologically with `pct_embargo`, and dropping conflicting points.
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
            # Extract test bounding map tracking test span explicitly
            test_times = self.t1.loc[X.index[test_indices]].copy()
            
            # Map forward using Embargo offset protecting against downstream autocorrelation
            for t0_test, t1_test in test_times.items():
                if pd.notna(t1_test) and t1_test in embargo_mapping.index:
                    test_times.loc[t0_test] = embargo_mapping.loc[t1_test]
                    
            # Set training candidates: observations entirely natively excluded from evaluating fold set
            train_candidates = self.t1.loc[X.index].drop(X.index[test_indices])
            
            # Execute purging envelope over candidate pool
            train_t1 = get_train_times(train_candidates, test_times)
            
            if len(train_t1) == 0:
                logger.warning("PurgedKFold Fold %d: All training features dropped under purging constraints.", fold_idx)
            
            logger.info("PurgedKFold Fold %d | Train: %d elements | Test: %d elements", fold_idx, len(train_t1), len(test_indices))
            
            # Translate explicit positional indices supporting generic scikit interoperability
            train_indices_array = X.index.get_indexer(train_t1.index)
            yield train_indices_array, test_indices
            fold_idx += 1

    def __repr__(self):
        return f"PurgedKFold(n_splits={self.n_splits}, pct_embargo={self.pct_embargo})"


class CombinatorialPurgedKFold:
    """Combinatorial Purged Cross-Validation (CPCV).

    MLDP Ch 12. Generates comprehensive combinations of evaluation folds yielding 
    `phi` orthogonal out-of-sample backtest paths. 
    """
    
    def __init__(self, n_splits=6, n_test_splits=2, t1=None, pct_embargo=0.01):
        """
        Initializes CPCV structural allocations.
        
        Args:
            n_splits (int): `N` static partitions mapping logical sets natively.
            n_test_splits (int): `k` specific group combinations extracted targeting test configurations.
            t1 (pd.Series): Target sequence limits supporting evaluation lengths.
            pct_embargo (float): Standard time mapping offset isolating correlation drift.
        """
        if t1 is None:
            raise ValueError("The parameter `t1` MUST be provided.")
        if n_test_splits >= n_splits / 2:
            raise ValueError("n_test_splits (k) must be strictly less than n_splits/2 (N/2).")
            
        self.n_splits = n_splits
        self.n_test_splits = n_test_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        
        # Calculate distinct evaluations supporting orthogonal mappings
        self.phi = math.comb(n_splits, n_test_splits)
        phi_paths = int(self.phi * self.n_test_splits / self.n_splits)
        
        logger.info(
            "Initialized CPCV: N=%d, k=%d. "
            "Total combinations: C(%d, %d) = %d. "
            "Total structural phi paths C(N, k) * k / N = %d",
            n_splits, n_test_splits, n_splits, n_test_splits, self.phi, phi_paths
        )

    def split(self, X, y=None, groups=None):
        if not X.index.is_monotonic_increasing:
            raise ValueError("X.index must execute with chronological strictly monotonic bounds.")
        if not X.index.isin(self.t1.index).all():
            raise ValueError("All chronological X indices must align correctly onto provided t1 mapping ranges.")
            
        # Partition data indices contiguously without stochastic permutations
        total_len = len(X)
        group_size = total_len // self.n_splits
        groups_list = []
        for i in range(self.n_splits):
            start = i * group_size
            end = (i + 1) * group_size if i < self.n_splits - 1 else total_len
            groups_list.append(np.arange(start, end))
            
        # Establish combinations & generic array offset lookup maps
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        embargo_mapping = get_embargo_times(X.index, self.pct_embargo)
        
        for combo_idx, test_group_indices in enumerate(combinations):
            test_indices = np.concatenate([groups_list[i] for i in test_group_indices])
            train_group_indices = [i for i in range(self.n_splits) if i not in test_group_indices]
            train_candidates_indices = np.concatenate([groups_list[i] for i in train_group_indices])
            
            # Form contiguous boundaries supporting accurate index logic
            test_times = self.t1.loc[X.index[test_indices]].copy()
            
            # Assign contiguous embargo temporal shift isolating serial correlations
            for t0_test, t1_test in test_times.items():
                if pd.notna(t1_test) and t1_test in embargo_mapping.index:
                    test_times.loc[t0_test] = embargo_mapping.loc[t1_test]
                    
            # Explicit exclusion protocol triggering purging across validation logic matrix
            train_candidates = self.t1.loc[X.index[train_candidates_indices]]
            train_t1 = get_train_times(train_candidates, test_times)
            
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
        
        Yields discrete array-sets isolating temporal evaluations for exactly `phi` logical routes
        dictating `C(N-1, k-1)` completely exhaustive timeline mappings.

        Returns:
            List[List[Tuple[int, int]]]: 
                Logical collection mapping discrete lists of exactly length `N`.
                Each tuple expresses: (test_group_index, combination_index).
        """
        combinations = list(itertools.combinations(range(self.n_splits), self.n_test_splits))
        
        # Build allocations mapping each target testing interval to its compatible specific splits 
        inventory = {g: [] for g in range(self.n_splits)}
        for c_idx, combo in enumerate(combinations):
            for g in combo:
                inventory[g].append(c_idx)
                
        # Paths required matches MLDP theoretical boundary equation mapping constraints.
        num_paths = math.comb(self.n_splits - 1, self.n_test_splits - 1)
        
        paths = []
        for _ in range(num_paths):
            path = []
            for g in range(self.n_splits):
                # Retrieve chronological logic split explicitly capturing each element logically
                c_idx = inventory[g].pop(0)
                path.append((g, c_idx))
            paths.append(path)
            
        return paths

    def __repr__(self):
        return f"CombinatorialPurgedKFold(n_splits={self.n_splits}, n_test_splits={self.n_test_splits}, pct_embargo={self.pct_embargo})"