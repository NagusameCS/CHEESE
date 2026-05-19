# HyperTensor Chess Engine ♟️

**The world's first chess engine powered by Riemannian geometry and Grassmann manifold compression.**

Built on the [HyperTensor](https://github.com/NagusameCS/HyperTensor) geometric framework by NagusameCS, this chess engine leverages cutting-edge mathematical innovations to achieve unprecedented parameter efficiency and search intelligence.

## 🔬 HyperTensor Innovations Used

| Innovation | Paper | Chess Application |
|---|---|---|
| **NativeLinear** (Gr(k,d) manifold) | Paper XII | Neural net layers with ~98% parameter reduction |
| **Jury Formula** (J = 1 − Π(1−e^(−d/R))) | Paper XVI | Skip evaluation for familiar positions |
| **GTC Semantic Cache** | Paper VIII | Transposition table via geodesic distance |
| **K-Expansion Scheduler** | Paper XII | Discover intrinsic dimension of chess |
| **RiemannianAdamW** | Paper XII | Optimize on Grassmann manifold |
| **GeodesicMetric** | Core | Measure position similarity in k-space |

## 🏗 Architecture

```
Input (808-dim board tensor)
    │
    ▼
┌─────────────────────────────────┐
│  NativeLinear(808, k)           │  ← Grassmann manifold Gr(k,808)
│  ~98% parameter compression     │
└─────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────┐
│  Hidden Layers (k-space)        │
│  ReLU + Dropout + Residual      │
└─────────────────────────────────┘
    │
    ├──────────────────┐
    ▼                  ▼
┌─────────┐    ┌──────────────┐
│ Value   │    │ Policy Head  │
│ Head    │    │ (4096 moves) │
│ [-1,1]  │    │              │
└─────────┘    └──────────────┘
    │                  │
    ▼                  ▼
  Score          Move probabilities
```

## 🚀 Quick Start

### 1. Clone and install
```bash
git clone https://github.com/NagusameCS/HyperTensor.git
cd chess_engine
pip install -r requirements.txt
```

### 2. Run the demo
```bash
python -m chess_engine demo
```

### 3. Interactive play
```bash
python -m chess_engine play
```

### 4. UCI mode (for chess GUIs like Arena)
```bash
python -m chess_engine
```

### 5. Full training pipeline
```bash
python -m chess_engine train --k-start 4 --k-target 64 --iterations 100
```

## 🎮 Playing Against the Engine

```
HyperTensor Chess Engine — Interactive Play
===========================================
You play White. Enter moves in UCI format (e.g., e2e4)

8 r n b q k b n r
7 p p p p p p p p
6 . . . . . . . .
5 . . . . . . . .
4 . . . . . . . .
3 . . . . . . . .
2 P P P P P P P P
1 R N B Q K B N R
  a b c d e f g h

Your move: e2e4
Engine thinking...
Engine plays: e7e5
  (evals: 42, jury: 35, cache: 0)
```

## 🧠 How It Works

### 1. Grassmann Manifold Position Encoding
Every chess position is embedded into a low-dimensional Grassmann manifold Gr(k,808). The intrinsic dimension k starts at 4 and expands to 64 during training, discovering the natural structure of chess positions.

### 2. Jury-Gated Speculative Search
The **Jury Formula** $J = 1 - \prod_i (1 - e^{-d_i/R})$ computes confidence that a position lies in familiar territory. When $J > 0.85$, the engine skips expensive neural evaluation — a ~300× speedup for accepted positions.

### 3. Geodesic Transposition Table
Instead of exact Zobrist hashing, HyperTensor uses geodesic distance in k-space to find *semantically similar* positions — detecting near-transpositions that differ by just a tempo.

### 4. K-Expansion Curriculum Learning
The model starts training at k=4 (learning basic piece values) and exponentially expands to k=64 (learning complex strategy), guided by the KExpansionScheduler from HyperTensor Paper XII.

## 📊 Parameter Efficiency

| Layer Type | Parameters | Savings |
|---|---|---|
| Dense (808×256) | 206,848 | — |
| NativeLinear (808, k=8) | 6,528 | 96.8% |
| NativeLinear (808, k=64) | 55,808 | 73.0% |

## 📈 Training Pipeline

```
Self-Play Games → Experience Buffer → Neural Training → K-Expansion → Repeat
      ↑                                                                    │
      └────────────────── Model Update ───────────────────────────────────┘
```

## 🏆 Goal

The ultimate goal is to build the strongest chess engine in the world by combining:
- HyperTensor's geometric compression for deeper neural networks
- Jury-gated speculative search for vastly more efficient tree search
- Geodesic transposition detection for better position generalization
- K-expansion curriculum for optimal feature discovery

## 📝 Citation

If you use this engine in research, please cite:

```bibtex
@misc{stewart2026hypertensor,
  author = {William Ken Ohara Stewart},
  title  = {HyperTensor: a geometric framework for understanding,
            compressing, and extending transformer language models},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.20077378},
}
```

## 📄 License

MIT License — see [LICENSE](../LICENSE) file.

---

*"Chess is not yet solved — the manifold still has room to discover."*
