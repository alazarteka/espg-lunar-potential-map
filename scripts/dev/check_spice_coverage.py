#!/usr/bin/env python3
"""
Diagnostic script to check SPICE kernel coverage for December 1998.

This script will:
1. Load all SPICE kernels
2. Check coverage for specific dates in December 1998
3. Identify gaps in ephemeris data
"""

import logging
from datetime import datetime, timedelta

import spiceypy as spice

from src.potential_mapper.spice import load_spice_files

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_coverage_for_date(year, month, day, hour=12):
    """Check if SPICE data is available for a specific date."""
    dt = datetime(year, month, day, hour, 0, 0)
    et = spice.utc2et(dt.strftime("%Y-%m-%d %H:%M:%S"))
    
    try:
        # Try to get LP position relative to Moon (matching actual pipeline)
        state, _ = spice.spkpos("LUNAR PROSPECTOR", et, "J2000", "NONE", "MOON")
        
        # Try to get Sun position relative to LP (matching actual pipeline)
        sun_state, _ = spice.spkpos("SUN", et, "J2000", "NONE", "LUNAR PROSPECTOR")
        
        return True, f"✓ {dt.date()} OK"
    except Exception as e:
        error_msg = repr(e)  # Use repr to see full error
        if "SPKINSUFFDATA" in error_msg:
            return False, f"✗ {dt.date()} INSUFFICIENT DATA"
        elif "BADSUBSCRIPT" in error_msg:
            return False, f"✗ {dt.date()} BAD SUBSCRIPT (corrupted?)"
        else:
            # Print full error for first occurrence
            return False, f"✗ {dt.date()} ERROR: {type(e).__name__}"


def main():
    logger.info("Loading SPICE kernels...")
    load_spice_files()
    
    logger.info("\n" + "="*60)
    logger.info("Checking SPICE coverage for December 1998")
    logger.info("="*60 + "\n")
    
    # Check every day in December 1998
    failures = []
    successes = []
    first_error_printed = False
    
    for day in range(1, 32):
        try:
            success, msg = check_coverage_for_date(1998, 12, day)
            if success:
                logger.info(msg)
                successes.append(day)
            else:
                if not first_error_printed:
                    # Print first error in detail
                    dt = datetime(1998, 12, day, 12, 0, 0)
                    et = spice.utc2et(dt.strftime("%Y-%m-%d %H:%M:%S"))
                    try:
                        state, _ = spice.spkpos("LUNAR PROSPECTOR", et, "J2000", "NONE", "MOON")
                    except Exception as debug_e:
                        logger.error(f"Full error for Dec {day}: {debug_e}")
                        first_error_printed = True
                logger.info(msg)
                failures.append((day, msg))
        except Exception as e:
            msg = f"✗ 1998-12-{day:02d} EXCEPTION: {str(e)[:60]}"
            logger.error(msg)
            failures.append((day, msg))
    
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)
    logger.info(f"Success: {len(successes)}/31 days")
    logger.info(f"Failures: {len(failures)}/31 days")
    
    if failures:
        logger.info("\nFailed dates:")
        for day, msg in failures:
            logger.info(f"  {msg}")
    
    # Check kernel info
    logger.info("\n" + "="*60)
    logger.info("Loaded SPICE Kernels")
    logger.info("="*60)
    count = spice.ktotal("ALL")
    logger.info(f"Total loaded kernels: {count}")
    
    for i in range(count):
        file, filtyp, source, handle = spice.kdata(i, "ALL", 255, 255, 255)
        logger.info(f"  [{i}] {filtyp}: {file}")


if __name__ == "__main__":
    main()
