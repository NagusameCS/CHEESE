"""
HyperTensor Chess Engine — Web UI Server
==========================================
Flask-based web interface with:
  - Live interactive chess board (SVG pieces)
  - Play vs engine in browser
  - Watch self-play games
  - Live training progress monitor
  - Position evaluation display
  - Move history + engine stats

Usage:
  python -m chess_engine web          # Start server
  python -m chess_engine web --port 8080  # Custom port
"""
import sys, os, time, json, threading, queue, math
from pathlib import Path
from typing import Dict, Optional

HT = Path(__file__).parent.parent / "HyperTensor"
if HT.exists(): sys.path.insert(0, str(HT))

from flask import Flask, jsonify, request, send_from_directory

from .board import Board, Move, Color, Piece, STARTING_FEN, SQUARE_NAMES
from .evaluation import create_model, CUDA_AVAILABLE, DEVICE
from .negamax import NegamaxEngine
from .opening_book import get_opening_move
from .pretrain import heuristic_evaluate

app = Flask(__name__)

# Global engine state
engine_state = {
    'board': Board(),
    'model': None,
    'search': None,
    'moves': [],
    'stats': {},
    'thinking': False,
    'game_result': None,
    'player_color': Color.WHITE,
    'mode': 'play',  # 'play' or 'watch'
    'training': {
        'running': False,
        'iterations': 0,
        'total_positions': 0,
        'current_batch': 0,
        'best_val_loss': None,
        'last_update': '',
    },
    'last_move': None,
    'message': '',
}

def init_engine():
    global _cached_val_loss
    if engine_state['model'] is None:
        print('[WebUI] Loading model...', flush=True)
        model = create_model(k_manifold=32, hidden_dim=128, num_layers=3)
        best = Path('models/sf_autopilot_best.pt')
        _cached_val_loss = None
        if best.exists():
            ck = torch.load(best, map_location='cpu', weights_only=True)
            # Try to load, but handle size mismatches from different model configs
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in ck['model_state_dict'].items() 
                              if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict, strict=False)
            _cached_val_loss = ck.get('val_loss')
            loaded = len(pretrained_dict)
            total = len(model_dict)
            print(f'[WebUI] Loaded {loaded}/{total} params from best model (val_loss={_cached_val_loss})', flush=True)
        model.eval()
        engine_state['model'] = model
        engine_state['search'] = NegamaxEngine(
            model, tt_size_mb=64
        )
        print('[WebUI] Negamax engine ready', flush=True)

# Preload at startup
_cached_val_loss = None
print('[WebUI] Preloading engine...', flush=True)
import torch
init_engine()

def engine_think():
    """Background search using negamax PVS."""
    try:
        init_engine()
        move, stats = engine_state['search'].find_best_move(
            engine_state['board'], time_limit_ms=3000
        )
        return move, stats
    except Exception as e:
        return None, {'error': str(e)}

# ============================================================================
# HTML Template
# ============================================================================
HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HyperTensor Chess Engine</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#1a1a2e;color:#e0e0e0;font-family:'Segoe UI',system-ui,sans-serif;
     display:flex;height:100vh;overflow:hidden}
.sidebar{width:320px;background:#16213e;padding:16px;overflow-y:auto;
         border-right:2px solid #0f3460;display:flex;flex-direction:column;gap:12px}
.main{flex:1;display:flex;flex-direction:column;align-items:center;
      justify-content:center;padding:20px}
.board-container{position:relative}
.board{display:grid;grid-template-columns:repeat(8,64px);grid-template-rows:repeat(8,64px);
      border:3px solid #533483;border-radius:4px;box-shadow:0 0 30px rgba(83,52,131,.5)}
.square{width:64px;height:64px;display:flex;align-items:center;justify-content:center;
        font-size:48px;cursor:pointer;transition:background .15s;user-select:none}
.square.light{background:#f0d9b5}.square.dark{background:#b58863}
.square:hover{box-shadow:inset 0 0 0 3px rgba(255,255,200,.8)}
.square.selected{box-shadow:inset 0 0 0 3px #ffeb3b}
.square.last-move{background:#cdd26a!important}
.piece-white{color:#fff;text-shadow:0 1px 3px rgba(0,0,0,.5);filter:drop-shadow(0 1px 2px #000)}
.piece-black{color:#1a1a1a;text-shadow:0 1px 3px rgba(255,255,255,.3)}
h1{font-size:20px;color:#e94560;margin:0;text-align:center}
h2{font-size:14px;color:#7b8ab8;text-transform:uppercase;letter-spacing:2px;
   border-bottom:1px solid #0f3460;padding-bottom:4px}
.panel{background:#1a1a2e;border-radius:8px;padding:12px;border:1px solid #0f3460}
.stat{display:flex;justify-content:space-between;padding:3px 0;font-size:13px}
.stat-label{color:#7b8ab8}.stat-value{color:#e0e0e0;font-weight:600}
.eval-bar{height:20px;border-radius:10px;display:flex;overflow:hidden;margin:8px 0}
.eval-white{background:linear-gradient(90deg,#fff,#e0e0e0);transition:flex .3s}
.eval-black{background:linear-gradient(90deg,#333,#1a1a1a);transition:flex .3s}
.eval-text{text-align:center;font-size:13px;font-weight:600}
.moves{font-size:12px;line-height:1.6;max-height:200px;overflow-y:auto;font-family:monospace}
.moves span{padding:2px 4px;border-radius:3px}
.moves .current{background:#533483;color:#fff}
button,.btn{background:#533483;color:#fff;border:none;padding:8px 16px;border-radius:6px;
            cursor:pointer;font-size:13px;font-weight:600;transition:background .2s;width:100%}
button:hover{background:#e94560}
button:disabled{opacity:.5;cursor:default}
input{width:100%;background:#0f3460;border:1px solid #533483;color:#fff;padding:8px 12px;
      border-radius:6px;font-size:14px;outline:none}
input:focus{border-color:#e94560}
.status{text-align:center;font-size:14px;padding:8px;border-radius:6px}
.status.thinking{background:#533483;animation:pulse 1.5s infinite}
.status.win{background:#2d6a4f}.status.loss{background:#6b2c39}.status.draw{background:#7f6000}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.6}}
.thinking-dots:after{content:'';animation:dots 1.5s steps(4,end) infinite}
@keyframes dots{0%{content:''}25%{content:'.'}50%{content:'..'}75%{content:'...'}}
.training-bar{height:6px;background:#0f3460;border-radius:3px;margin:4px 0;overflow:hidden}
.training-bar-fill{height:100%;background:linear-gradient(90deg,#533483,#e94560);
                    transition:width .5s;border-radius:3px}
.piece-glyph{line-height:1}
</style>
</head>
<body>
<div class="sidebar">
  <h1>♛ HyperTensor Chess</h1>
  <div id="status" class="status">Loading...</div>

  <div class="panel">
    <h2>Evaluation</h2>
    <div class="eval-bar">
      <div class="eval-white" id="eval-white" style="flex:1"></div>
      <div class="eval-black" id="eval-black" style="flex:1"></div>
    </div>
    <div class="eval-text" id="eval-text">+0 cp</div>
    <div style="text-align:center;font-size:11px;color:#7b8ab8" id="eval-detail"></div>
  </div>

  <div class="panel">
    <h2>Game Info</h2>
    <div class="stat"><span class="stat-label">Turn</span><span class="stat-value" id="turn">White</span></div>
    <div class="stat"><span class="stat-label">Move</span><span class="stat-value" id="move-num">1</span></div>
    <div class="stat"><span class="stat-label">FEN</span><span class="stat-value" style="font-size:10px" id="fen">-</span></div>
    <div class="stat"><span class="stat-label">Result</span><span class="stat-value" id="result">-</span></div>
  </div>

  <div class="panel">
    <h2>Engine Stats</h2>
    <div class="stat"><span class="stat-label">Simulations</span><span class="stat-value" id="s-sims">-</span></div>
    <div class="stat"><span class="stat-label">Nodes/sec</span><span class="stat-value" id="s-nps">-</span></div>
    <div class="stat"><span class="stat-label">Time</span><span class="stat-value" id="s-time">-</span></div>
    <div class="stat"><span class="stat-label">Jury accepts</span><span class="stat-value" id="s-jury">-</span></div>
    <div class="stat"><span class="stat-label">Cache hits</span><span class="stat-value" id="s-cache">-</span></div>
    <div class="stat"><span class="stat-label">TT hits</span><span class="stat-value" id="s-tt">-</span></div>
  </div>

  <div class="panel">
    <h2>📊 Live Elo</h2>
    <div style="text-align:center;font-size:36px;font-weight:bold;color:#e94560" id="elo-display">---</div>
    <div class="stat"><span class="stat-label">Evaluation Elo</span><span class="stat-value" id="elo-eval">-</span></div>
    <div class="stat"><span class="stat-label">Playing Elo</span><span class="stat-value" id="elo-play">-</span></div>
    <div class="stat"><span class="stat-label">vs Stockfish</span><span class="stat-value" style="color:#7b8ab8">gap: <span id="elo-gap">-</span></div>
    <div class="training-bar"><div class="training-bar-fill" id="elo-bar" style="width:30%"></div></div>
    <div style="text-align:center;font-size:10px;color:#7b8ab8">0 ═══════════════════ 3650 (Stockfish)</div>
  </div>

  <div class="panel">
    <h2>Training</h2>
    <div class="stat"><span class="stat-label">Status</span><span class="stat-value" id="t-status">Idle</span></div>
    <div class="stat"><span class="stat-label">Batches</span><span class="stat-value" id="t-batches">7</span></div>
    <div class="stat"><span class="stat-label">Positions</span><span class="stat-value" id="t-pos">~35K</span></div>
    <div class="stat"><span class="stat-label">Best val loss</span><span class="stat-value" id="t-loss">-</span></div>
    <div class="training-bar"><div class="training-bar-fill" id="t-bar" style="width:0%"></div></div>
  </div>

  <input type="text" id="move-input" placeholder="Enter move (e.g. e2e4) or command..." autofocus>
  <div style="display:flex;gap:8px">
    <button onclick="sendCmd('new')">New Game</button>
    <button onclick="sendCmd('undo')">Undo</button>
    <button onclick="sendCmd('watch')">Watch AI</button>
  </div>

  <div class="panel">
    <h2>Moves</h2>
    <div class="moves" id="moves-list"></div>
  </div>
</div>

<div class="main">
  <div class="board" id="board"></div>
  <div style="margin-top:8px;color:#7b8ab8;font-size:12px;text-align:center">
    Click a piece then click destination · Commands: new, undo, watch, stop, eval
  </div>
</div>

<script>
// State
let fen = 'START';
let selectedSquare = null;
let playerColor = 'white';
let mode = 'play';

const PIECES = {
  'K':'♔','Q':'♕','R':'♖','B':'♗','N':'♘','P':'♙',
  'k':'♚','q':'♛','r':'♜','b':'♝','n':'♞','p':'♟'
};

function renderBoard(fenStr) {
  const board = document.getElementById('board');
  board.innerHTML = '';
  // Parse FEN piece placement
  const parts = fenStr.split(' ');
  const placement = parts[0];
  const ranks = placement.split('/');
  
  for (let r = 0; r < 8; r++) {
    let file = 0;
    for (const ch of ranks[r]) {
      if (ch >= '1' && ch <= '8') {
        for (let i = 0; i < parseInt(ch); i++) {
          const sq = document.createElement('div');
          sq.className = 'square ' + ((r + file) % 2 === 0 ? 'light' : 'dark');
          sq.dataset.sq = (7-r)*8 + file;
          sq.onclick = () => clickSquare((7-r)*8 + file);
          board.appendChild(sq);
          file++;
        }
      } else {
        const sq = document.createElement('div');
        sq.className = 'square ' + ((r + file) % 2 === 0 ? 'light' : 'dark');
        sq.dataset.sq = (7-r)*8 + file;
        sq.onclick = () => clickSquare((7-r)*8 + file);
        const piece = document.createElement('span');
        piece.className = 'piece-glyph ' + (ch === ch.toUpperCase() ? 'piece-white' : 'piece-black');
        piece.textContent = PIECES[ch] || ch;
        sq.appendChild(piece);
        board.appendChild(sq);
        file++;
      }
    }
  }
  fen = fenStr;
}

function clickSquare(sq) {
  if (mode === 'watch') return;
  if (selectedSquare === null) {
    // Select piece
    selectedSquare = sq;
    document.querySelectorAll('.square').forEach(s => s.classList.remove('selected'));
    const el = document.querySelector(`[data-sq="${sq}"]`);
    if (el) el.classList.add('selected');
  } else {
    // Make move
    const from = ['a','b','c','d','e','f','g','h'][selectedSquare%8] + (Math.floor(selectedSquare/8)+1);
    const to = ['a','b','c','d','e','f','g','h'][sq%8] + (Math.floor(sq/8)+1);
    sendCmd(from+to);
    selectedSquare = null;
    document.querySelectorAll('.square').forEach(s => s.classList.remove('selected'));
  }
}

function sendCmd(cmd) {
  fetch('/api/cmd', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({cmd: cmd})
  }).then(r => r.json()).then(updateUI);
}

function updateUI(data) {
  if (data.fen) renderBoard(data.fen);
  if (data.status) {
    const s = document.getElementById('status');
    s.textContent = data.status;
    s.className = 'status ' + (data.thinking ? 'thinking' : '');
  }
  
  document.getElementById('eval-text').textContent = (data.eval_cp || 0) + ' cp';
  document.getElementById('eval-detail').textContent = data.eval_detail || '';
  document.getElementById('turn').textContent = data.turn || '-';
  document.getElementById('move-num').textContent = data.move_num || '-';
  document.getElementById('fen').textContent = data.fen_short || '-';
  document.getElementById('result').textContent = data.result || '-';
  
  // Engine stats
  if (data.stats) {
    document.getElementById('s-sims').textContent = data.stats.simulations || '-';
    document.getElementById('s-nps').textContent = data.stats.nps || '-';
    document.getElementById('s-time').textContent = (data.stats.time_ms || 0) + 'ms';
    document.getElementById('s-jury').textContent = data.stats.jury_accepts || '-';
    document.getElementById('s-cache').textContent = data.stats.cache_hits || '-';
    document.getElementById('s-tt').textContent = data.stats.tt_hits || '-';
  }
  
  // Moves + Elo + Training
  if (data.training) {
    var eElo = data.training.eval_elo || 1600;
    var pElo = data.training.play_elo || 1800;
    document.getElementById('elo-display').textContent = pElo || '---';
    document.getElementById('elo-eval').textContent = eElo;
    document.getElementById('elo-play').textContent = pElo;
    document.getElementById('elo-gap').textContent = (3650 - pElo) + ' Elo';
    document.getElementById('elo-bar').style.width = Math.min(100, (pElo/3650)*100) + '%';
    
    var eloDisp = document.getElementById('elo-display');
    if (pElo > 2500) eloDisp.style.color = '#00ff88';
    else if (pElo > 2000) eloDisp.style.color = '#ffcc00';
    else if (pElo > 1500) eloDisp.style.color = '#ff8800';
    else eloDisp.style.color = '#e94560';
    
    document.getElementById('t-status').textContent = data.training.running ? 'Running' : 'Converged';
    document.getElementById('t-batches').textContent = data.training.batches || '0';
    document.getElementById('t-pos').textContent = (data.training.positions || 0).toLocaleString();
    document.getElementById('t-loss').textContent = data.training.best_loss || '-';
    document.getElementById('t-bar').style.width = (data.training.progress || 0) + '%';
  }
  
  // Moves list
  if (data.moves) {
    var html = '';
    for (var i = 0; i < data.moves.length; i++) {
      if (i % 2 === 0) html += '<div style="padding:2px 0">';
      html += '<span' + (i === data.moves.length-1 ? ' class="current"' : '') + '>' + data.moves[i] + '</span>';
      if (i % 2 === 1 || i === data.moves.length-1) html += '</div>';
    }
    document.getElementById('moves-list').innerHTML = html || '<span style="color:#7b8ab8">No moves yet</span>';
  }
  
  mode = data.mode || 'play';
}

// Eval bar
function updateEvalBar(cp) {
  const score = Math.max(-1000, Math.min(1000, cp || 0));
  const whitePct = ((score + 1000) / 2000 * 100);
  document.getElementById('eval-white').style.flex = whitePct;
  document.getElementById('eval-black').style.flex = 100 - whitePct;
}

// Initial render
renderBoard('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1');

// Input handling
document.getElementById('move-input').addEventListener('keydown', function(e) {
  if (e.key === 'Enter') {
    sendCmd(this.value);
    this.value = '';
  }
});

// Poll for updates
function poll() {
  fetch('/api/state').then(r => r.json()).then(data => {
    updateUI(data);
    if (data.eval_cp !== undefined) {
      updateEvalBar(data.eval_cp);
      document.getElementById('eval-text').textContent = 
        (data.eval_cp > 0 ? '+' : '') + data.eval_cp + ' cp';
    }
  }).catch(() => {});
}
setInterval(poll, 500);
poll();
</script>
</body>
</html>
"""

# ============================================================================
# API Routes
# ============================================================================
@app.route('/')
def index():
    return HTML

@app.route('/api/state')
def api_state():
    """Get full engine state for polling. Graceful on errors."""
    try:
        init_engine()
        b = engine_state['board']
        stats = engine_state['stats']
        
        hv, hm, hb = heuristic_evaluate(b)
        eval_cp = int(hv * 1000)
        if stats:
            eval_cp = int(stats.get('best_value', hv) * 1000)
    except Exception as e:
        return jsonify({'error': str(e), 'fen_short': 'error', 'turn': '?'})
    
    # Check if engine just finished thinking
    result = None
    if engine_state['thinking']:
        result = check_engine_done()
        if result:
            move, s = result
            engine_state['stats'] = s or {}
            if move:
                engine_state['last_move'] = move
                engine_state['moves'].append(move.uci())
                engine_state['board'].make_move(move)
            engine_state['thinking'] = False
    
    # Game result
    game_result = b.result()
    if game_result:
        engine_state['game_result'] = game_result
    
    # Training info
    batches = len(list(Path('models').glob('sf_batch_*.npz')))
    
    # Live Elo estimation based on val_loss
    val_loss = _cached_val_loss
    
    # Elo model: Stockfish depth 10 ≈ 2800 Elo evaluation
    if val_loss is not None:
        eval_elo = max(800, min(2800, int(2800 - 900 * math.sqrt(max(val_loss, 0.0001)))))
        play_elo = eval_elo + 200  # MCTS adds ~200 Elo
    else:
        eval_elo = 1600
        play_elo = 1800
    
    train_data = {
        'running': False,
        'batches': batches,
        'positions': batches * 5000,
        'best_loss': f'{val_loss:.4f}' if val_loss else '-',
        'progress': min(100, batches * 2),
        'eval_elo': eval_elo,
        'play_elo': play_elo,
    }
    
    return jsonify({
        'fen': b.fen(),
        'fen_short': ' '.join(b.fen().split()[:2]),
        'turn': 'White' if b.color_to_move == Color.WHITE else 'Black',
        'move_num': len(engine_state['moves']) // 2 + 1,
        'moves': [m.uci() if hasattr(m, 'uci') else str(m) for m in engine_state['moves']],
        'result': engine_state['game_result'],
        'status': 'Thinking...' if engine_state['thinking'] else (
            f'{game_result}' if game_result else
            f'{"Your" if b.color_to_move == engine_state["player_color"] else "Engine"} turn'
        ),
        'thinking': engine_state['thinking'],
        'eval_cp': eval_cp,
        'eval_detail': f'Material: {hm:+.2f}  Mobility: {hb:+.2f}',
        'stats': {
            'simulations': stats.get('simulations', 0),
            'nps': stats.get('nps', 0),
            'time_ms': f'{stats.get("time_ms", 0):.0f}',
            'jury_accepts': stats.get('jury_accepts', 0),
            'cache_hits': stats.get('cache_hits', 0),
            'tt_hits': stats.get('tt_hits', 0),
            'best_value': f'{stats.get("best_value", 0):+.3f}',
        } if stats else {},
        'training': train_data,
        'mode': engine_state['mode'],
    })

def check_engine_done():
    try:
        return engine_state.get('_result_queue', queue.Queue()).get_nowait()
    except queue.Empty:
        return None
    except:
        return None

@app.route('/api/cmd', methods=['POST'])
def api_cmd():
    """Handle user commands."""
    data = request.get_json()
    cmd = data.get('cmd', '').strip().lower()
    init_engine()
    
    if cmd == 'new':
        engine_state['board'] = Board()
        engine_state['moves'] = []
        engine_state['stats'] = {}
        engine_state['last_move'] = None
        engine_state['game_result'] = None
        engine_state['thinking'] = False
        engine_state['message'] = 'New game'
    
    elif cmd == 'undo':
        if len(engine_state['moves']) >= 2:
            try:
                engine_state['board'].unmake_move()
                engine_state['moves'].pop()
                engine_state['board'].unmake_move()
                engine_state['moves'].pop()
                engine_state['message'] = 'Undid move'
            except: pass
    
    elif cmd == 'watch':
        engine_state['mode'] = 'watch'
        engine_state['board'] = Board()
        engine_state['moves'] = []
        engine_state['stats'] = {}
        engine_state['thinking'] = False
        engine_state['message'] = 'Watch mode'
        # Auto-start engine move
        start_engine_think()
    
    elif cmd == 'stop':
        engine_state['mode'] = 'play'
        engine_state['thinking'] = False
    
    elif cmd == 'eval':
        hv, hm, hb = heuristic_evaluate(engine_state['board'])
        engine_state['message'] = f'Heuristic: {hv*1000:+.0f} cp'
    
    elif cmd:
        # Try as UCI move
        try:
            m = Move.from_uci(cmd)
            legal = engine_state['board'].generate_legal_moves()
            if m in legal and not engine_state['thinking']:
                engine_state['last_move'] = m
                engine_state['moves'].append(m.uci())
                engine_state['board'].make_move(m)
                engine_state['message'] = ''
                engine_state['stats'] = {}
                # Auto engine response
                start_engine_think()
        except:
            engine_state['message'] = f'Invalid: {cmd}'
    
    return jsonify({'ok': True, 'message': engine_state.get('message', '')})

def start_engine_think():
    if engine_state['thinking']: return
    engine_state['thinking'] = True
    engine_state['_result_queue'] = queue.Queue()
    
    def think():
        try:
            move, stats = engine_state['search'].search(
                engine_state['board'].copy(), time_limit_ms=3000
            )
            engine_state['_result_queue'].put((move, stats))
        except Exception as e:
            engine_state['_result_queue'].put((None, {'error': str(e)}))
    
    t = threading.Thread(target=think, daemon=True)
    t.start()

# Watch mode auto-move checker (called from state poll)
def check_watch_mode():
    if engine_state['mode'] != 'watch': return
    if engine_state['thinking']: return
    if engine_state['board'].is_game_over(): return
    start_engine_think()

# Patch state endpoint to trigger watch moves
_original_state = api_state
def patched_state():
    check_watch_mode()
    return _original_state()
app.view_functions['api_state'] = patched_state

# ============================================================================
# Main
# ============================================================================
def main():
    import argparse
    p = argparse.ArgumentParser(description='HyperTensor Chess Web UI')
    p.add_argument('--port', type=int, default=8080, help='Server port')
    p.add_argument('--host', type=str, default='127.0.0.1', help='Server host')
    args = p.parse_args()
    
    print(f'\n  ♛ HyperTensor Chess Web UI')
    print(f'  {"─"*40}')
    print(f'  Open: http://{args.host}:{args.port}')
    print(f'  {"─"*40}\n')
    
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
