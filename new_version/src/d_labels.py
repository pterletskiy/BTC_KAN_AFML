"""
4_labels.py — Triple-Barrier Method and Sample Weighting logic for MFW pipeline.

This module sits downstream of 3_econometrics.py and upstream of 5_cv.py.
It implements Marcos López de Prado's Triple-Barrier Method (TBM) for 
time-series labeling, along with concurrent sample weight generation
based on overlapping holding periods and absolute returns.
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
def mpPandasObj(func, pdObj, numThreads=None, **kargs):
    """
    Parallelize a pandas operation over a partitioned list of atoms.

    Parameters
    ----------
    func : callable
        Worker function to dispatch. Must accept `molecule` as its last positional
        argument (or explicitly capture other args from pdObj).
    pdObj : tuple
        Tuple where `pdObj[0]` is the list/index of atoms to partition into molecules.
        Remaining elements are passed as positional arguments to `func` before the molecule.
    numThreads : int, optional
        Number of parallel workers. Defaults to max(1, cpu_count() - 1).
    **kargs : 
        Additional keyword arguments to pass to `func`.

    Returns
    -------
    pd.DataFrame or pd.Series
        Concatenated results across all dispatch workers.
    """
    if numThreads is None:
        numThreads = max(1, mp.cpu_count() - 1)

    atoms = pdObj[0]
    if len(atoms) == 0:
        # Return empty immediately if no tasks
        return pd.DataFrame()

    numThreads = max(1, min(numThreads, len(atoms)))
    step = int(np.ceil(len(atoms) / numThreads))
    molecules = [atoms[i:i + step] for i in range(0, len(atoms), step)]
    
    results = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=numThreads) as executor:
        futures = []
        for mol in molecules:
            # Reconstruct arguments: pdObj[1:] + [molecule]
            args = list(pdObj[1:]) + [mol]
            futures.append(executor.submit(func, *args, **kargs))
            
        for future in futures:
            try:
                # keep implicit order of submission by iterating futures
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
    # Create daily resampled series. Use strictly the last price of the day.
    df0 = close.resample('D').last().dropna()
    
    # Calculate daily log returns
    df0 = np.log(df0 / df0.shift(1))
    
    # Exponential moving standard deviation
    df0 = df0.ewm(span=span0).std()
    
    # Forward-fill daily volatility estimates to original timestamps
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
        
        # Valid path slice based on t1
        if pd.isna(t1_val):
            close_slice = close.loc[t0:]
        else:
            close_slice = close.loc[t0:t1_val]
            
        if close_slice.empty:
            continue
            
        # Log return path relative to entry t0
        path = np.log(close_slice / close.loc[t0])
        trgt = events.loc[t0, 'trgt']
        
        pt = ptSl[0] * trgt if ptSl[0] > 0 else np.inf
        sl = -ptSl[1] * trgt if ptSl[1] > 0 else -np.inf
        
        # Upper (profit taking) touch
        pt_idx = path[path >= pt].index
        pt_touch = pt_idx[0] if len(pt_idx) > 0 else pd.NaT
        
        # Lower (stop loss) touch
        sl_idx = path[path <= sl].index
        sl_touch = sl_idx[0] if len(sl_idx) > 0 else pd.NaT
        
        out.loc[t0, 'sl'] = sl_touch
        out.loc[t0, 'pt'] = pt_touch
        out.loc[t0, 't1'] = t1_val  # Note: t1 is evaluated as the third limit
        
    return out


def getEvents(
    close: pd.Series,
    tEvents: pd.DatetimeIndex,
    ptSl: list,
    trgt: pd.Series,
    minRet: float,
    numThreads: int,
    t1=False,
    side: pd.Series = None
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

    Returns
    -------
    pd.DataFrame
        Events DataFrame with ['t1', 'trgt', 'side'], indexed by t0.
    """
    # 1. Filter timestamps
    tEvents = tEvents[tEvents.isin(trgt.index)]
    trgt_aligned = trgt.loc[tEvents]
    
    # Apply minimum return filter
    valid_events = trgt_aligned[trgt_aligned >= minRet].index
    if valid_events.empty:
        return pd.DataFrame(columns=['t1', 'trgt', 'side'])
        
    trgt_valid = trgt_aligned.loc[valid_events]
    
    # 2. Prepare vertical barrier
    if t1 is False:
        t1_series = pd.Series(pd.NaT, index=valid_events)
    else:
        t1_series = t1.loc[valid_events]
        
    events = pd.DataFrame({'t1': t1_series, 'trgt': trgt_valid}, index=valid_events)
    
    # 3. Handle side prediction
    if side is None:
        events['side'] = 1.0
    else:
        events['side'] = side.loc[valid_events]
        
    # 4. Dispatch horizontal barriers
    pdObj = (events.index, close, events, ptSl)
    df0 = mpPandasObj(applyPtSlOnT1, pdObj, numThreads=numThreads)
    
    # 5. Consolidate: first touch out of [sl, pt, t1]
    # Note: min() skips NaNs automatically, finding the earliest time.
    events['t1'] = df0.min(axis=1, skipna=True)
    events = events.dropna(subset=['t1'])
    
    return events


def getBins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """Assign classification labels {-1, 0, 1} to events.

    Parameters
    ----------
    events : pd.DataFrame
        Output of `getEvents` featuring columns 't1', 'trgt', 'side'.
    close : pd.Series
        Price series.

    Returns
    -------
    pd.DataFrame
        Labeling outputs DataFrame featuring 'ret' and 'bin'.
    """
    events_valid = events.dropna(subset=['t1']).copy()
    out = pd.DataFrame(index=events_valid.index)
    
    # Safely obtain prices
    t1_prices = close.loc[events_valid['t1']].values
    t0_prices = close.loc[events_valid.index].values
    
    # Log return over the lifespan [t0, t1]
    ret = np.log(t1_prices / t0_prices)
    out['ret'] = ret
    
    # Meta-labeling logic or standard directional
    if 'side' in events_valid.columns and (events_valid['side'] != 1.0).any():
        # Meta-labeling: bin=1 if side matches log-directional sign exactly
        out['bin'] = np.where((out['ret'] * events_valid['side']) > 0, 1, 0)
    else:
        # Standard target: strictly mapping sign of ret
        out['bin'] = np.sign(out['ret'])
        
        # Mask near-zero returns to avoid labeling microscopic drift
        # Enforcing boundary condition: |ret| must surpass 50% of the dynamic target threshold.
        mask_tiny = np.abs(out['ret']) < (events_valid['trgt'].values * 0.5)
        out.loc[mask_tiny, 'bin'] = 0
        
    return out

# ==============================================================================
# 3. SAMPLE WEIGHTS & UNIQUENESS
# ==============================================================================
def mpNumCoEvents(closeIdx: pd.DatetimeIndex, t1: pd.Series, molecule: list) -> pd.Series:
    """Worker function: counts active (concurrent) labels at each bar.

    Parameters
    ----------
    closeIdx : pd.DatetimeIndex
        Base index to count concurrent events over.
    t1 : pd.Series
        Mapping of t0 to t1.
    molecule : list
        Timestamps of events to process.

    Returns
    -------
    pd.Series
        Counts of concurrent active labels, stripped of zero-values.
    """
    t1_mol = t1.loc[molecule]
    counts = pd.Series(0.0, index=closeIdx)
    
    for t0, t1_val in t1_mol.items():
        if pd.isna(t1_val):
            continue
        counts.loc[t0:t1_val] += 1.0
        
    # Return sparse series to optimize IPC sizes and aggregation
    return counts[counts > 0]


def mpSampleTW(t1: pd.Series, numCoEvents: pd.Series, molecule: list) -> pd.Series:
    """Worker function: computes average uniqueness over an event's lifespan.

    Parameters
    ----------
    t1 : pd.Series
        Mapping of t0 to t1.
    numCoEvents : pd.Series
        System-wide counts of concurrent events (no zeroes).
    molecule : list
        Events to process.

    Returns
    -------
    pd.Series
        Average uniqueness ranging (0, 1].
    """
    out = pd.Series(index=molecule, dtype=float)
    
    for t0 in molecule:
        t1_val = t1.loc[t0]
        if pd.isna(t1_val):
            continue
            
        uniqueness = 1.0 / numCoEvents.loc[t0:t1_val]
        out.loc[t0] = uniqueness.mean()
        
    return out


def getAvgUniqueness(t1: pd.Series, numThreads: int = None) -> pd.Series:
    """Orchestrator: Generates average uniqueness for full set of labels.

    Parameters
    ----------
    t1 : pd.Series
        Set of event endpoints.
    numThreads : int, optional
        Number of workers.

    Returns
    -------
    pd.Series
        Full dataset average uniqueness indicators.
    """
    closeIdx = t1.index
    pdObj1 = (t1.index, closeIdx, t1)
    
    df0 = mpPandasObj(mpNumCoEvents, pdObj1, numThreads=numThreads)
    numCoEvents = df0.groupby(level=0).sum() if not df0.empty else pd.Series()
    
    # Align to closeIdx & floor at 1 to evade div/0
    numCoEvents = numCoEvents.reindex(closeIdx).fillna(0).clip(lower=1.0)
    
    pdObj2 = (t1.index, t1, numCoEvents)
    avgU = mpPandasObj(mpSampleTW, pdObj2, numThreads=numThreads)
    
    return avgU


def mpSampleW(t1: pd.Series, numCoEvents: pd.Series, close: pd.Series, molecule: list) -> pd.Series:
    """Worker function: generates absolute final sample weight combining average uniqueness and logs returns.

    Parameters
    ----------
    t1 : pd.Series
        Mapping of t0 to t1.
    numCoEvents : pd.Series
        System-wide counts of concurrent events.
    close : pd.Series
        Price series.
    molecule : list
        Events to process.

    Returns
    -------
    pd.Series
        Aggregated unscaled sample weight component.
    """
    out = pd.Series(index=molecule, dtype=float)
    for t0 in molecule:
        t1_val = t1.loc[t0]
        if pd.isna(t1_val):
            continue
            
        ret = np.log(close.loc[t1_val] / close.loc[t0])
        uniqueness = 1.0 / numCoEvents.loc[t0:t1_val]
        out.loc[t0] = abs(ret) * uniqueness.mean()
        
    return out


def getSampleWeights(t1: pd.Series, close: pd.Series, numThreads: int = None) -> pd.Series:
    """Orchestrator: Generates balanced sample weight vector combining correlation discount & log returns.

    Parameters
    ----------
    t1 : pd.Series
        Mapping of t0 to t1.
    close : pd.Series
        Price series.
    numThreads : int, optional
        Number of parallel workers.

    Returns
    -------
    pd.Series
        Normalized sample weights matching N density.
    """
    closeIdx = t1.index
    pdObj_co = (t1.index, closeIdx, t1)
    
    # 1. Parallelize CoEvent counting
    df0 = mpPandasObj(mpNumCoEvents, pdObj_co, numThreads=numThreads)
    numCoEvents = df0.groupby(level=0).sum() if not df0.empty else pd.Series()
    numCoEvents = numCoEvents.reindex(closeIdx).fillna(0).clip(lower=1.0)
    
    # 2. Parallelize specific Sample Weight mapping
    pdObj_sw = (t1.index, t1, numCoEvents, close)
    weights = mpPandasObj(mpSampleW, pdObj_sw, numThreads=numThreads)
    
    if weights.empty:
        return weights
        
    if weights.isna().any() or np.isinf(weights).any():
        raise ValueError("Sample Weights contain NaNs or Inifinities. Validate price series integrity.")
        
    # 3. Normalization logic matching MLDP standard weighting (matches unity sum density over N samples)
    weights = weights * (len(weights) / weights.sum())
    return weights

# ==============================================================================
# 4. PRIMARY API ORCHESTRATOR
# ==============================================================================
def run_labels(
    close: pd.Series,
    tEvents: pd.DatetimeIndex,
    numDays: int = 5,
    ptSl: list = [1, 1],
    minRet: float = 0.005,
    span0: int = 100,
    numThreads: int = None,
    saveInterim: bool = True,
    interim_path: str = "data/interim/"
) -> dict:
    """Main orchestration routine spanning volatility targets to sampling weights computation.

    Parameters
    ----------
    close : pd.Series
        Price matrix series strictly evaluated to contain raw level logs.
    tEvents : pd.DatetimeIndex
        Base temporal vectors.
    numDays : int
        Time span scalar converting directly into fixed timeline.
    ptSl : list
        Constraint targets multipliers setting profit and loss targets.
    minRet : float
        Min boundary limits controlling density cutoff.
    span0 : int
        Rolling threshold parameter.
    numThreads : int, optional
        Resource density controller dictating subprocesses.
    saveInterim : bool
        True forces write persistence to file.
    interim_path : str
        Write-out namespace directory standard output.

    Returns
    -------
    dict
        Combined payload containing exactly 'events', 'bins', & 'sampleWeights'.
    """
    if numThreads is None:
        numThreads = max(1, mp.cpu_count() - 1)
        
    # Safety Check: Drop unknown index events
    unknowns = tEvents[~tEvents.isin(close.index)]
    if not unknowns.empty:
        logger.warning("Dropped %d events from tEvents not present in close index.", len(unknowns))
        tEvents = tEvents[tEvents.isin(close.index)]
    
    pre_count = len(tEvents)
        
    # 1. Volatility setup
    trgt = getDailyVol(close, span0=span0)
    
    # 2. Vertical explicit layout processing 
    t1 = pd.Series(pd.NaT, index=tEvents)
    tdelta = pd.Timedelta(days=numDays)
    
    # Naive vertical allocation logic supporting gap-invariant temporal offsets
    for t0 in tEvents:
        valid_post = close.index[close.index >= t0 + tdelta]
        if not valid_post.empty:
            t1.loc[t0] = valid_post[0]
        else:
            t1.loc[t0] = close.index[-1]
            
    # 3. Executing Core Matrix Generation Limits (TBM logic application)
    events = getEvents(
        close=close,
        tEvents=tEvents,
        ptSl=ptSl,
        trgt=trgt,
        minRet=minRet,
        numThreads=numThreads,
        t1=t1,
        side=None
    )
    post_count = len(events)
    
    # 4. Labeling & Meta Binning Resolution 
    bins = getBins(events, close)
    
    # 5. Extracting Core Weights Mapping Logic Space
    sampleWeights = getSampleWeights(events['t1'], close, numThreads=numThreads)
    
    # Calculate standalone Uniqueness for isolated logs rendering
    avgU_df = getAvgUniqueness(events['t1'], numThreads=numThreads)
    
    logger.info("LABELING STATS: Initial events: %d | Active Post-minRet: %d", pre_count, post_count)
    logger.info("LABELING STATS: Avg Uniqueness -> Mean: %.4f, Std: %.4f", avgU_df.mean(), avgU_df.std())
    logger.info("LABELING STATS: Sample Weights -> Mean: %.4f, Std: %.4f", sampleWeights.mean(), sampleWeights.std())
    
    if saveInterim:
        out_dir = Path(interim_path)
        out_dir.mkdir(parents=True, exist_ok=True)
        events.to_frame() if isinstance(events, pd.Series) else events.to_parquet(out_dir / "events.parquet")
        bins.to_frame() if isinstance(bins, pd.Series) else bins.to_parquet(out_dir / "bins.parquet")
        sampleWeights.to_frame(name="weight").to_parquet(out_dir / "sampleWeights.parquet")
        logger.info("Persisted labeling logic to `%s`", out_dir)

    return {
        'events': events,
        'bins': bins,
        'sampleWeights': sampleWeights
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info("4_labels.py module initialized.")