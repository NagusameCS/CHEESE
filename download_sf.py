"""Find and download Stockfish binary."""
import urllib.request, json, os, subprocess, sys

def try_download(url, name):
    try:
        print(f'  {name}...', end=' ', flush=True)
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        resp = urllib.request.urlopen(req, timeout=30)
        data = resp.read()
        if len(data) > 2000000:
            with open('stockfish.exe', 'wb') as f:
                f.write(data)
            print(f'OK ({len(data):,} bytes)')
            return True
        print(f'too small ({len(data)})')
    except Exception as e:
        print(f'FAIL ({type(e).__name__})')
    return False

# Query GitHub API for correct download URL
print('Querying GitHub API for latest Stockfish release...')
try:
    req = urllib.request.Request(
        'https://api.github.com/repos/official-stockfish/Stockfish/releases/latest',
        headers={'User-Agent': 'Mozilla/5.0', 'Accept': 'application/vnd.github+json'}
    )
    resp = urllib.request.urlopen(req, timeout=15)
    data = json.loads(resp.read())
    tag = data.get('tag_name', '')
    print(f'Latest tag: {tag}')
    
    for asset in data.get('assets', []):
        name = asset['name'].lower()
        if 'windows' in name and ('avx2' in name or 'x86-64' in name) and name.endswith('.exe'):
            if try_download(asset['browser_download_url'], asset['name']):
                break
    else:
        print('No Windows asset found, trying fallback URLs...')
        for url in [
            'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64-avx2.exe',
            'https://github.com/official-stockfish/Stockfish/releases/latest/download/stockfish-windows-x86-64.exe',
        ]:
            if try_download(url, url.split('/')[-1]):
                break
except Exception as e:
    print(f'GitHub API failed: {e}')

# Test binary
if os.path.exists('stockfish.exe') and os.path.getsize('stockfish.exe') > 2000000:
    print('\nTesting Stockfish...')
    try:
        r = subprocess.run(['stockfish.exe', 'uci'], capture_output=True, text=True, timeout=10)
        for line in r.stdout.split('\n'):
            if 'id name' in line:
                print(line.strip())
                break
        r2 = subprocess.run(['stockfish.exe', 'bench'], capture_output=True, text=True, timeout=30)
        for line in r2.stdout.split('\n'):
            if 'Nodes/second' in line or 'nps' in line.lower():
                print(line.strip())
    except Exception as e:
        print(f'Test failed: {e}')
else:
    print('\nCould not download Stockfish automatically.')
    print('Please download manually from: https://stockfishchess.org/download/')
    print('Place stockfish.exe in this directory.')
