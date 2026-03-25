"""
4_labels.py — Triple-Barrier Method and Sample Weighting logic for MFW pipeline.

This module sits downstream of 3_econometrics.py and upstream of 5_cv.py.
It implements Marcos López de Prado's Triple-Barrier Method (TBM) for 
time-series labeling, along with concurrent sample weight generation
based on overlapping holding periods and absolute returns.

CHANGELOG:
- Added `getTimeDecay` integrating AFML piecewise-linear decay algorithms cleanly.
- Implemented `seqBootstrap` enforcing dynamic average uniqueness probabilities resolving overlaps.
- Removed arbitrary `0` masks in `getBins` actively routing towards explicit label dropping. 
- Eradicated mutable default arguments gracefully universally natively.
- Shifted default scaling architecture onto `ThreadPoolExecutor` safely inherently mapping memory.
- Completely vectorized `searchsorted` vertical timeline scanning optimally!
- Rectified buggy interim parquets mappings explicitly resolving outputs correctly seamlessly.
- Configured dynamic label imbalance logs highlighting `< 10%` boundary degradation.
- Fixed literal typos structurally gracefully natively!
- Redefined `mpPandasObj` stripping brittle tuple dispatching limits properly smoothly.
- Injected strict AFML references advising `class_weight='balanced'` downstream practically. 
"""

import concurrent.futures
import logging
import math
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ==============================================================================
# MULTIPROCESSING ENGINE
# ==============================================================================
def mpPandasObj(func, atoms, *args, numThreads=None, useProcesses=False, **kwargs):
    """
    Parallelize a pandas operation over a partitioned list of atoms.

    Parameters
    ----------
    func : callable
        Worker function to dispatch. Must accept `molecule` as its last positional
        argument (or explicitly capture other args).
    atoms : list-like
        The index of items to partition into molecules.
    *args : 
        Additional positional arguments passed to `func` before the molecule.
    numThreads : int, optional
        Number of parallel workers. Defaults to max(1, cpu_count() - 1).
    useProcesses : bool
        If True, utilizes ProcessPoolExecutor (heavy). Defaults to ThreadPoolExecutor.
    **kwargs : 
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    pd.DataFrame or pd.Series
        Concatenated results across all dispatch workers.
    """
    if numThreads is None:
        numThreads = max(1, mp.cpu_count() - 1)

    if len(atoms) == 0:
        return pd.DataFrame()

    numThreads = max(1, min(numThreads, len(atoms)))
    step = int(np.ceil(len(atoms) / numThreads))
    molecules = [atoms[i:i + step] for i in range(0, len(atoms), step)]
    
    results = []
    ExecutorClass = concurrent.futures.ProcessPoolExecutor if useProcesses else concurrent.futures.ThreadPoolExecutor
    
    with ExecutorClass(max_workers=numThreads) as executor:
        futures = []
        for mol in molecules:
            mol_args = list(args) + [mol]
            futures.append(executor.submit(func, *mol_args, **kwargs))
            
        for future in futures:
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("mpPandasObj worker failed: %s", e)
                raise

    if not results:
        return pd.DataFrame()
        
    if isinstance(results[0], (pd.Series, pd.DataFrame)):
        out = pd.concat(results)
        return out
    return results

# ==============================================================================
# 1. VOLATILITY ESTIMATION
# ==============================================================================
def getDailyVol(close: pd.Series, span0: int = 100) -> pd.Series:
    """Computes the daily volatility estimate at each intraday timestamp.
    
    Uses exponential moving standard deviation of daily log returns,
    forward-filled to the original timestamp frequency.

    Parameters
    ----------
    close : pd.Series
        Price series with a strictly monotonic pd.DatetimeIndex.
    span0 : int
        Span for the EWMA standard deviation (default 100).

    Returns
    -------
    pd.Series
        Daily volatility track assigned to intraday timestamps, forward-filled.
        Named 'dailyVol'.
    """
    df0 = close.resample('D').last().dropna()
    df0 = np.log(df0 / df0.shift(1))
    df0 = df0.ewm(span=span0).std()
    
    df0 = df0.reindex(close.index, method='ffill')
    df0.name = 'dailyVol'
    return df0

# ==============================================================================
# 2. TRIPLE-BARRIER LABELING
# ==============================================================================
def applyPtSlOnT1(close: pd.Series, events: pd.DataFrame, ptSl: list, molecule: list) -> pd.DataFrame:
    """Worker function: evaluates horizontal barriers for a molecule of events.
    
    Determines the exact timestamp at which the first barrier (profit-taking,
    stop-loss, or the pre-defined vertical time barrier) is touched.

    Parameters
    ----------
    close : pd.Series
        Price series.
    events : pd.DataFrame
        Slices of target returns ('trgt') and vertical barriers ('t1').
    ptSl : list
        Multipliers [pt_multiplier, sl_multiplier]. A multiplier of 0 disables that barrier.
    molecule : list
        Index labels (subset of events.index) to process.

    Returns
    -------
    pd.DataFrame
        First cross times in columns ['sl', 'pt', 't1'].
    """
    out = pd.DataFrame(columns=['sl', 'pt', 't1'], index=molecule)
    
    for t0 in molecule:
        t1_val = events.loc[t0, 't1']
        
        if pd.isna(t1_val):
            close_slice = close.loc[t0:]
        else:
            close_slice = close.loc[t0:t1_val]
            
        if close_slice.empty:
            continue
            
        path = np.log(close_slice / close.loc[t0])
        trgt = events.loc[t0, 'trgt']
        
        pt = ptSl[0] * trgt if ptSl[0] > 0 else np.inf
        sl = -ptSl[1] * trgt if ptSl[1] > 0 else -np.inf
        
        pt_idx = path[path >= pt].index
        pt_touch = pt_idx[0] if len(pt_idx) > 0 else pd.NaT
        
        sl_idx = path[path <= sl].index
        sl_touch = sl_idx[0] if len(sl_idx) > 0 else pd.NaT
        
        out.loc[t0, 'sl'] = sl_touch
        out.loc[t0, 'pt'] = pt_touch
        out.loc[t0, 't1'] = t1_val  
        
    return out


def getEvents(
    close: pd.Series,
    tEvents: pd.DatetimeIndex,
    ptSl: list,
    trgt: pd.Series,
    minRet: float,
    numThreads: int,
    t1=False,
    side: pd.Series = None,
    useProcesses: bool = False
) -> pd.DataFrame:
    """Orchestrates the Triple-Barrier Method labeling logic.
    
    Maps each seed timestamp to its barrier-touch time and target.

    Parameters
    ----------
    close : pd.Series
        Price series.
    tEvents : pd.DatetimeIndex
        Seed timestamps to evaluate.
    ptSl : list
        Profit-taking and stop-loss multipliers.
    trgt : pd.Series
        Dynamic volatility targets, aligned with `close`.
    minRet : float
        Minimum target return required to place a barrier.
    numThreads : int
        Number of parallel workers.
    t1 : pd.Series or bool, optional
        Pre-defined vertical barrier timestamps indexed by `tEvents`.
    side : pd.Series, optional
        Primary model predictions for asymmetric metalabeling.
    useProcesses : bool
        If True, dispatch executes natively across heavy Process pools.

    Returns
    -------
    pd.DataFrame
        Events DataFrame with ['t1', 'trgt', 'side'], indexed by t0.
    """
    tEvents = tEvents[tEvents.isin(trgt.index)]
    trgt_aligned = trgt.loc[tEvents]
    
    valid_events = trgt_aligned[trgt_aligned >= minRet].index
    if valid_events.empty:
        return pd.DataFrame(columns=['t1', 'trgt', 'side'])
        
    trgt_valid = trgt_aligned.loc[valid_events]
    
    if t1 is False:
        t1_series = pd.Series(pd.NaT, index=valid_events)
    else:
        t1_series = t1.loc[valid_events]
        
    events = pd.DataFrame({'t1': t1_series, 'trgt': trgt_valid}, index=valid_events)
    
    if side is None:
        events['side'] = 1.0
    else:
        events['side'] = side.loc[valid_events]
        
    df0 = mpPandasObj(
        applyPtSlOnT1, events.index, close, events, ptSl, 
        numThreads=numThreads, useProcesses=useProcesses
    )
    
    events['t1'] = df0.min(axis=1, skipna=True)
    events = events.dropna(subset=['t1'])
    
    return events


def getBins(events: pd.DataFrame, close: pd.Series, dropLabels: bool = True, minPctLabel: float = 0.0) -> pd.DataFrame:
    """Assign classification labels {-1, 0, 1} to events.
    
    AFML Chapter 4.8 Note: Class imbalances can still persist. Downstream
    classifiers must use `class_weight='balanced'` or `'balanced_subsample'`
    to robustly correct residual imbalance.

    Parameters
    ----------
    events : pd.DataFrame
        Output of `getEvents` featuring columns 't1', 'trgt', 'side'.
    close : pd.Series
        Price series.
    dropLabels : bool
        If True, cleanly removes arrays beneath exact bounds. Defaults True supporting AFML. 
    minPctLabel : float
        Bounded multiplier tracking threshold percentage parameters seamlessly.

    Returns
    -------
    pd.DataFrame
        Labeling outputs DataFrame featuring 'ret' and 'bin'.
    """
    events_valid = events.dropna(subset=['t1']).copy()
    out = pd.DataFrame(index=events_valid.index)
    
    t1_prices = close.loc[events_valid['t1']].values
    t0_prices = close.loc[events_valid.index].values
    
    ret = np.log(t1_prices / t0_prices)
    out['ret'] = ret
    
    if 'side' in events_valid.columns and (events_valid['side'] != 1.0).any():
        out['bin'] = np.where((out['ret'] * events_valid['side']) > 0, 1, 0)
    else:
        out['bin'] = np.sign(out['ret'])
        
        mask_tiny = np.abs(out['ret']) < (events_valid['trgt'].values * minPctLabel)
        
        if dropLabels:
            out = out[~mask_tiny]
        else:
            out.loc[mask_tiny, 'bin'] = 0
            logger.warning("[getBins] dropLabels=False deviates from AFML strictly. "
                           "Return-attributed zero labels may systematically receive underweighted importance.")
            
    return out

# ==============================================================================
# 3. SAMPLE WEIGHTS, UNIQUENESS & BOOTSTRAP
# ==============================================================================
def mpNumCoEvents(closeIdx: pd.DatetimeIndex, t1: pd.Series, molecule: list) -> pd.Series:
    """Worker function: counts active (concurrent) labels at each bar."""
    t1_mol = t1.loc[molecule]
    counts = pd.Series(0.0, index=closeIdx)
    
    for t0, t1_val in t1_mol.items():
        if pd.isna(t1_val):
            continue
        counts.loc[t0:t1_val] += 1.0
        
    return counts[counts > 0]


def mpSampleTW(t1: pd.Series, numCoEvents: pd.Series, molecule: list) -> pd.Series:
    """Worker function: computes average uniqueness over an event's lifespan."""
    out = pd.Series(index=molecule, dtype=float)
    
    for t0 in molecule:
        t1_val = t1.loc[t0]
        if pd.isna(t1_val):
            continue
            
        uniqueness = 1.0 / numCoEvents.loc[t0:t1_val]
        out.loc[t0] = uniqueness.mean()
        
    return out


def getAvgUniqueness(t1: pd.Series, numThreads: int = None, useProcesses: bool = False) -> pd.Series:
    """Orchestrator: Generates average uniqueness for full set of labels."""
    closeIdx = t1.index
    df0 = mpPandasObj(
        mpNumCoEvents, t1.index, closeIdx, t1, 
        numThreads=numThreads, useProcesses=useProcesses
    )
    
    numCoEvents = df0.groupby(level=0).sum() if not df0.empty else pd.Series()
    numCoEvents = numCoEvents.reindex(closeIdx).fillna(0).clip(lower=1.0)
    
    avgU = mpPandasObj(
        mpSampleTW, t1.index, t1, numCoEvents, 
        numThreads=numThreads, useProcesses=useProcesses
    )
    return avgU


def mpSampleW(t1: pd.Series, numCoEvents: pd.Series, close: pd.Series, molecule: list) -> pd.Series:
    """Worker function: generates absolute final sample weight combining average uniqueness and logs returns."""
    out = pd.Series(index=molecule, dtype=float)
    for t0 in molecule:
        t1_val = t1.loc[t0]
        if pd.isna(t1_val):
            continue
            
        ret = np.log(close.loc[t1_val] / close.loc[t0])
        uniqueness = 1.0 / numCoEvents.loc[t0:t1_val]
        out.loc[t0] = abs(ret) * uniqueness.mean()
        
    return out


def getTimeDecay(tW: pd.Series, clfLastW: float = 1.0) -> pd.Series:
    """Implement piecewise-linear time decay on sample weights (AFML Snippet 4.11).
    
    Parameters
    ----------
    tW : pd.Series
        Output from getSampleWeights (return-attributed * average uniqueness).
    clfLastW : float, optional
        Decay factor:
        c = 1 : No decay.
        0 < c < 1 : Weights decay linearly but remain positive.
        c = 0 : Weights converge linearly to 0 for oldest observations.
        c < 0 : Oldest portion of observations receives zero weight.
        
    Returns
    -------
    pd.Series
        Time-decayed sample weights.
    """
    clfW = tW.sort_index().cumsum()
    if clfLastW >= 0:
        slope = (1.0 - clfLastW) / clfW.iloc[-1]
    else:
        slope = 1.0 / ((clfLastW + 1.0) * clfW.iloc[-1])
        
    decay = 1.0 - (slope * (clfW.iloc[-1] - clfW))
    decay.loc[decay < 0] = 0
    return decay


def getSampleWeights(
    t1: pd.Series, 
    close: pd.Series, 
    numThreads: int = None, 
    decay_c: float = 1.0, 
    useProcesses: bool = False
) -> pd.Series:
    """Orchestrator: Generates balanced sample weight vector combining correlation discount & log returns."""
    closeIdx = t1.index
    df0 = mpPandasObj(
        mpNumCoEvents, t1.index, closeIdx, t1, 
        numThreads=numThreads, useProcesses=useProcesses
    )
    
    numCoEvents = df0.groupby(level=0).sum() if not df0.empty else pd.Series()
    numCoEvents = numCoEvents.reindex(closeIdx).fillna(0).clip(lower=1.0)
    
    weights = mpPandasObj(
        mpSampleW, t1.index, t1, numCoEvents, close, 
        numThreads=numThreads, useProcesses=useProcesses
    )
    
    if weights.empty:
        return weights
        
    if weights.isna().any() or np.isinf(weights).any():
        raise ValueError("Sample Weights contain NaNs or Infinities. Validate price series integrity.")
        
    weights = weights * (len(weights) / weights.sum())
    
    if decay_c != 1.0:
        decay_factors = getTimeDecay(weights, clfLastW=decay_c)
        weights = weights * decay_factors
        weights = weights * (len(weights) / weights.sum())
        
    return weights


def getIndMatrix(barIx: pd.DatetimeIndex, t1: pd.Series) -> pd.DataFrame:
    """Build the binary indicator matrix mapping each label's lifespan to the index."""
    indM = pd.DataFrame(0.0, index=barIx, columns=range(t1.shape[0]))
    for i, (t0, t1_val) in enumerate(t1.items()):
        if pd.isna(t1_val):
            continue
        indM.loc[t0:t1_val, i] = 1.0
    return indM


def seqBootstrap(t1: pd.Series, numSamples: int = None, random_state: int = None) -> list:
    """Draw samples sequentially proportional to their average uniqueness (AFML Chapter 4.5)."""
    if numSamples is None:
        numSamples = t1.shape[0]
    if random_state is not None:
        np.random.seed(random_state)
        
    phi = []
    # Index limits evaluating exact timeline
    barIx = t1.index.union(t1.dropna().values).sort_values().drop_duplicates()
    indM = getIndMatrix(barIx, t1)
    
    # Pre-extract to numpy for fast O(N^2) evaluation
    indM_np = indM.values 
    N, M = indM_np.shape
    
    indM_sum = np.zeros(N)
    
    for i in range(numSamples):
        if i == 0:
            prob = np.ones(M) / M
        else:
            prob = np.zeros(M)
            for j in range(M):
                cand = indM_np[:, j]
                denom = indM_sum + cand
                active = cand > 0
                if active.any():
                    prob[j] = np.mean(cand[active] / denom[active])
                else:
                    prob[j] = 0.0
            prob = prob / prob.sum()
            
        choice = np.random.choice(M, p=prob)
        phi.append(choice)
        indM_sum += indM_np[:, choice]
        
    return [t1.index[i] for i in phi]

# ==============================================================================
# 4. PRIMARY API ORCHESTRATOR
# ==============================================================================
def run_labels(
    close: pd.Series,
    tEvents: pd.DatetimeIndex,
    numDays: int = 5,
    ptSl: list = None,
    minRet: float = 0.005,
    minPctLabel: float = 0.0,
    dropLabels: bool = True,
    decay_c: float = 1.0,
    span0: int = 100,
    numThreads: int = None,
    useProcesses: bool = False,
    saveInterim: bool = True,
    interim_path: str = "data/interim/"
) -> dict:
    """Main orchestration routine spanning volatility targets to sampling weights computation."""
    if numThreads is None:
        numThreads = max(1, mp.cpu_count() - 1)
        
    if ptSl is None:
        ptSl = [1, 1]
        
    unknowns = tEvents[~tEvents.isin(close.index)]
    if not unknowns.empty:
        logger.warning("Dropped %d events from tEvents not present in close index.", len(unknowns))
        tEvents = tEvents[tEvents.isin(close.index)]
    
    pre_count = len(tEvents)
        
    # 1. Volatility setup
    trgt = getDailyVol(close, span0=span0)
    
    # 2. Vectorized Vertical Explicit Bounds scanning
    tdelta = pd.Timedelta(days=numDays)
    t1_values = tEvents + tdelta
    
    idx_positions = close.index.searchsorted(t1_values, side='left')
    idx_positions = np.clip(idx_positions, 0, len(close.index) - 1)
    t1 = pd.Series(close.index[idx_positions], index=tEvents)
            
    # 3. Executing Core Matrix Generation Limits
    events = getEvents(
        close=close,
        tEvents=tEvents,
        ptSl=ptSl,
        trgt=trgt,
        minRet=minRet,
        numThreads=numThreads,
        t1=t1,
        side=None,
        useProcesses=useProcesses
    )
    
    # 4. Labeling Resolution strictly extracting bounds
    bins = getBins(events, close, dropLabels=dropLabels, minPctLabel=minPctLabel)
    
    # Crucially propagate any bounds dropped directly upstream syncing states
    events = events.loc[bins.index]
    post_count = len(events)
    
    # 5. Extracting Core Weights Mapping Logic Space + Bootstrap
    sampleWeights = getSampleWeights(
        events['t1'], close, 
        numThreads=numThreads, decay_c=decay_c, useProcesses=useProcesses
    )
    
    # Safely resolving sequences 
    seqBootstrapIdx = seqBootstrap(events['t1'], numSamples=len(bins))
    
    avgU_df = getAvgUniqueness(events['t1'], numThreads=numThreads, useProcesses=useProcesses)
    
    # Logs evaluating explicit constraints bounds naturally cleanly gracefully 
    label_counts = bins['bin'].value_counts().sort_index()
    logger.info("LABELING STATS: Initial events: %d | Active Post-minRet/Drops: %d", pre_count, post_count)
    logger.info("LABELING STATS: Label distribution:\n%s", label_counts.to_string())
    
    total = len(bins)
    for label, count in label_counts.items():
        if float(count) / total < 0.10:
            logger.warning("Class %s represents less than 10%% of the total samples (%.2f%%).", label, 100 * count / total)
            
    logger.info("LABELING STATS: Avg Uniqueness -> Mean: %.4f, Std: %.4f", avgU_df.mean(), avgU_df.std())
    logger.info("LABELING STATS: Sample Weights -> Mean: %.4f, Std: %.4f", sampleWeights.mean(), sampleWeights.std())
    
    if saveInterim:
        out_dir = Path(interim_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        
        if isinstance(events, pd.Series):
            events.to_frame().to_parquet(out_dir / "events.parquet")
        else:
            events.to_parquet(out_dir / "events.parquet")
            
        if isinstance(bins, pd.Series):
            bins.to_frame().to_parquet(out_dir / "bins.parquet")
        else:
            bins.to_parquet(out_dir / "bins.parquet")
            
        sampleWeights.to_frame(name="weight").to_parquet(out_dir / "sampleWeights.parquet")
        logger.info("Persisted labeling logic to `%s`", out_dir)

    return {
        'events': events,
        'bins': bins,
        'sampleWeights': sampleWeights,
        'seqBootstrapIdx': seqBootstrapIdx
    }


# PEP 8 aliases for backward compatibility dynamically
get_daily_vol = getDailyVol
get_time_decay = getTimeDecay
seq_bootstrap = seqBootstrap
get_ind_matrix = getIndMatrix
apply_pt_sl_on_t1 = applyPtSlOnT1
get_events = getEvents
get_bins = getBins
get_avg_uniqueness = getAvgUniqueness
get_sample_weights = getSampleWeights
mp_pandas_obj = mpPandasObj


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("4_labels.py module initialized.")