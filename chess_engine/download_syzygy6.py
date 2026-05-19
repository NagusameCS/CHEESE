"""
HyperTensor Chess — 6-Piece Syzygy Tablebase Downloader
=========================================================
Downloads 6-piece WDL+DTZ tablebases (~150GB total).
Provides perfect endgame play for positions with ≤6 pieces.

Source: https://tablebase.lichess.ovh/tables/standard/6/

Usage:
  python chess_engine/download_syzygy6.py --output syzygy/6/ --wdl-only
  python chess_engine/download_syzygy6.py --output syzygy/6/ --all
"""

import urllib.request
import os
import sys
import argparse
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


# 6-piece WDL files (~80GB) - essential for perfect play
WDL_FILES_6 = [
    # K + 5 pieces
    'KQQQQvK.rtbw', 'KQQQRvK.rtbw', 'KQQRRvK.rtbw', 'KQRRRvK.rtbw',
    'KRRRRvK.rtbw', 'KQQBBvK.rtbw', 'KQQNNvK.rtbw', 'KQRBBvK.rtbw',
    'KQRNNvK.rtbw', 'KRRBBvK.rtbw', 'KRRNNvK.rtbw',
    # Common endgame piece combinations
    'KQvKRP.rtbw', 'KQvKBP.rtbw', 'KQvKNP.rtbw', 'KQvKQP.rtbw',
    'KQvKRR.rtbw', 'KQvKBB.rtbw', 'KQvKNN.rtbw',
    'KRvKBP.rtbw', 'KRvKNP.rtbw', 'KRvKQP.rtbw',
    'KRvKRR.rtbw', 'KRvKBB.rtbw', 'KRvKNN.rtbw',
    'KBvKNP.rtbw', 'KBvKBP.rtbw', 'KNvKBP.rtbw',
    'KPvKBP.rtbw', 'KPvKNP.rtbw',
    # Pawn endgames
    'KPPvKP.rtbw', 'KPPvKB.rtbw', 'KPPvKN.rtbw', 'KPPvKR.rtbw',
    'KBPvKP.rtbw', 'KNPvKP.rtbw', 'KRPvKP.rtbw', 'KQPvKP.rtbw',
    # Major piece endgames
    'KRRvKQ.rtbw', 'KQRvKR.rtbw', 'KQRvKQ.rtbw',
    'KQBvKQ.rtbw', 'KQNvKQ.rtbw', 'KRBvKR.rtbw', 'KRNvKR.rtbw',
    'KBBvKN.rtbw', 'KBBvKB.rtbw', 'KNNvKN.rtbw',
    # Queen vs pieces
    'KQvKBN.rtbw', 'KQvKBB.rtbw', 'KQvKNN.rtbw', 'KQvKQ.rtbw',
    # Rook endgames
    'KRvKQ.rtbw', 'KRvKR.rtbw',
    # 3 minors vs nothing
    'KBBvK.rtbw', 'KBNvK.rtbw', 'KNNvK.rtbw',
]

# 6-piece DTZ files (~70GB) - needed for optimal play
DTZ_FILES_6 = [
    f.replace('.rtbw', '.rtbz') for f in WDL_FILES_6
]

BASE_URL = 'https://tablebase.lichess.ovh/tables/standard/6/'


def download_file(url: str, output_path: str, retries: int = 3) -> bool:
    """Download a single file with retries."""
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, output_path)
            return True
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f'  FAILED: {os.path.basename(output_path)}: {e}')
                return False
    return False


def download_tablebases(output_dir: str, files: list, max_workers: int = 4):
    """Download multiple tablebase files in parallel."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Check which files already exist
    to_download = []
    already_have = 0
    for f in files:
        path = os.path.join(output_dir, f)
        if os.path.exists(path):
            already_have += 1
        else:
            to_download.append(f)
    
    if already_have > 0:
        print(f'{already_have} files already present, {len(to_download)} to download')
    
    if not to_download:
        print('All files already downloaded!')
        return
    
    total = len(to_download)
    completed = 0
    total_size_mb = 0
    
    def download_one(filename):
        nonlocal completed, total_size_mb
        url = BASE_URL + filename
        path = os.path.join(output_dir, filename)
        success = download_file(url, path)
        if success:
            size = os.path.getsize(path) / (1024 * 1024)
            completed += 1
            total_size_mb += size
            if completed % 10 == 0 or completed == total:
                print(f'  [{completed}/{total}] {size:.0f}MB {filename} '
                      f'({total_size_mb:.0f}MB total)')
        return success
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_one, f): f for f in to_download}
        for future in as_completed(futures):
            future.result()
    
    print(f'Downloaded {completed}/{total} files ({total_size_mb:.0f}MB)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download 6-piece Syzygy tablebases')
    parser.add_argument('--output', default='syzygy/6', help='Output directory')
    parser.add_argument('--all', action='store_true', help='Download WDL + DTZ')
    parser.add_argument('--wdl-only', action='store_true', help='Only WDL files')
    parser.add_argument('--dtz-only', action='store_true', help='Only DTZ files')
    parser.add_argument('--workers', type=int, default=4, help='Parallel downloads')
    
    args = parser.parse_args()
    
    print('=' * 60)
    print('HyperTensor Chess — 6-Piece Syzygy Tablebase Downloader')
    print('Source: tablebase.lichess.ovh')
    print(f'Output: {args.output}')
    print('=' * 60)
    
    if args.wdl_only:
        print(f'\nDownloading {len(WDL_FILES_6)} WDL files (~80GB)...')
        download_tablebases(args.output, WDL_FILES_6, args.workers)
    elif args.dtz_only:
        print(f'\nDownloading {len(DTZ_FILES_6)} DTZ files (~70GB)...')
        download_tablebases(args.output, DTZ_FILES_6, args.workers)
    else:
        print(f'\nDownloading ALL 6-piece tablebases...')
        print(f'WDL: {len(WDL_FILES_6)} files (~80GB)')
        download_tablebases(args.output, WDL_FILES_6, args.workers)
        print(f'\nDTZ: {len(DTZ_FILES_6)} files (~70GB)')
        download_tablebases(args.output, DTZ_FILES_6, args.workers)
    
    # Verification
    wdl_count = len([f for f in WDL_FILES_6 
                     if os.path.exists(os.path.join(args.output, f))])
    dtz_count = len([f for f in DTZ_FILES_6 
                     if os.path.exists(os.path.join(args.output, f))])
    
    print(f'\nSyzygy 6-piece status:')
    print(f'  WDL: {wdl_count}/{len(WDL_FILES_6)} files')
    print(f'  DTZ: {dtz_count}/{len(DTZ_FILES_6)} files')
    
    # Calculate total size
    total_size = 0
    for f in WDL_FILES_6 + DTZ_FILES_6:
        path = os.path.join(args.output, f)
        if os.path.exists(path):
            total_size += os.path.getsize(path)
    
    print(f'  Total: {total_size / (1024**3):.1f} GB')
    print(f'\nConfigure in engine: probe_path = "{args.output}"')
