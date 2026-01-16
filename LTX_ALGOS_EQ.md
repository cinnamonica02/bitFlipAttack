# Formal Algorithm Specifications and Mathematical Formulations

This document provides paper-ready LaTeX equations and algorithm pseudocode for the Bit Flip Attack implementation.

## Table of Contents

1. [Mathematical Notation](#mathematical-notation)
2. [Core Equations](#core-equations)
3. [Algorithm Pseudocode](#algorithm-pseudocode)

---

## Mathematical Notation

### Model and Dataset

- $\mathcal{M}_\theta$: Neural network model with parameters $\theta = \{\mathbf{W}^{(1)}, \mathbf{W}^{(2)}, \ldots, \mathbf{W}^{(L)}\}$
- $\mathbf{W}^{(\ell)} \in \mathbb{R}^{d_{\ell} \times d_{\ell-1}}$: Weight matrix of layer $\ell$
- $w_{ij}^{(\ell)}$: Individual weight parameter at layer $\ell$, position $(i,j)$
- $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^{N}$: Dataset with $N$ samples
- $\mathcal{L}(\theta; \mathcal{D})$: Loss function over dataset $\mathcal{D}$

### Attack Parameters

- $B$: Maximum number of bits to flip (budget constraint)
- $\mathcal{C}$: Set of candidate bit positions $\mathcal{C} = \{c_1, c_2, \ldots, c_K\}$
- $c_k = (\ell, i, j, b)$: Bit candidate at layer $\ell$, parameter $(i,j)$, bit position $b$
- $\mathcal{S} \subseteq \mathcal{C}$: Selected subset of bits to flip, $|\mathcal{S}| \leq B$
- $\theta^*$: Perturbed model parameters after bit flips

### Metrics

- $\text{Acc}(\mathcal{M}_\theta)$: Model accuracy on test set
- $\text{ASR}(\mathcal{M}_\theta, y_t)$: Attack Success Rate for target class $y_t$
- $\tau_{\text{acc}}$: Accuracy threshold (minimum acceptable accuracy)

---

## Core Equations

### 1. Hybrid Sensitivity Metric

The sensitivity of a parameter combines magnitude and gradient information:

```latex
\begin{equation}
\text{sensitivity}(w_{ij}^{(\ell)}) = \alpha \cdot \|\nabla_{w_{ij}^{(\ell)}} \mathcal{L}\|_2 + (1 - \alpha) \cdot \|w_{ij}^{(\ell)}\|_2
\end{equation}

\text{where:}
\begin{aligned}
w_{ij}^{(\ell)} &: \text{weight parameter at layer } \ell, \text{ position } (i,j) \\
\nabla_{w_{ij}^{(\ell)}} \mathcal{L} &: \text{gradient of loss with respect to } w_{ij}^{(\ell)} \\
\alpha &\in [0,1] : \text{balance parameter (default: } \alpha = 0.5\text{)} \\
\alpha = 0 &: \text{pure magnitude-based sensitivity} \\
\alpha = 1 &: \text{pure gradient-based sensitivity} \\
\alpha = 0.5 &: \text{balanced hybrid approach}
\end{aligned}
```

### 2. Layer Vulnerability Score

The vulnerability of layer $\ell$ is measured by simulated attack impact:

```latex
V^{(\ell)} = \mathcal{L}(\theta^{(\ell)}_{\text{perturbed}}; \mathcal{D}) - \mathcal{L}(\theta; \mathcal{D})
```

where $\theta^{(\ell)}_{\text{perturbed}}$ is obtained by:

```latex
w_{ij}^{(\ell)} \leftarrow -w_{ij}^{(\ell)}, \quad \forall (i,j) \in \text{TopK}(S(\mathbf{W}^{(\ell)}), k)
```

with $k = \max(1, \lfloor 0.001 \times |\mathbf{W}^{(\ell)}| \rfloor)$ (0.1% of layer parameters).

### 3. Bit Position Priority

For IEEE 754 floating-point representation:

```latex
w = (-1)^s \times 2^{e - \text{bias}} \times (1 + m)
```

where:
- $s$: Sign bit (position 31 for FP32, 15 for FP16)
- $e$: Exponent bits (positions 30-23 for FP32, 14-10 for FP16)
- $m$: Mantissa bits (positions 22-0 for FP32, 9-0 for FP16)

**Impact Ranking:**

```latex
\text{Impact}(b) = \begin{cases}
    \infty & \text{if } b = b_{\text{sign}} \quad \text{(inverts value)} \\
    2^{|b - b_{\text{exp\_mid}}|} & \text{if } b \in \text{Exponent} \quad \text{(scales magnitude)} \\
    2^{-|b|} & \text{if } b \in \text{Mantissa} \quad \text{(precision change)}
\end{cases}
```

### 4. Bit Flip Operation

The bit flip operation on parameter $w$ at bit position $b$:

```latex
\text{FlipBit}(w, b) = \text{Float}(\text{Int}(w) \oplus 2^b)
```

where:
- $\text{Int}(w)$: Converts float to unsigned integer bit representation
- $\oplus$: XOR operation
- $\text{Float}(\cdot)$: Converts unsigned integer back to float

### 5. Attack Success Rate (ASR)

**Targeted Attack:**

```latex
\text{ASR}_{\text{targeted}}(\mathcal{M}_\theta, y_t) = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{test}}} \mathbb{1}[\mathcal{M}_\theta(\mathbf{x}) = y_t]
```

**Untargeted Attack:**

```latex
\text{ASR}_{\text{untargeted}}(\mathcal{M}_\theta) = \frac{1}{|\mathcal{D}_{\text{test}}|} \sum_{(\mathbf{x}, y) \in \mathcal{D}_{\text{test}}} \mathbb{1}[\mathcal{M}_\theta(\mathbf{x}) \neq y]
```

### 6. Fitness Function (Genetic Algorithm)

The fitness of a bit flip combination $\mathcal{S}$ is:

```latex
F(\mathcal{S}) = \begin{cases}
    \text{ASR}(\mathcal{M}_{\theta^*(\mathcal{S})}, y_t) & \text{if } \text{Acc}(\mathcal{M}_{\theta^*(\mathcal{S})}) \geq \tau_{\text{acc}} \\
    \text{ASR}(\mathcal{M}_{\theta^*(\mathcal{S})}, y_t) - \lambda \cdot (\tau_{\text{acc}} - \text{Acc}(\mathcal{M}_{\theta^*(\mathcal{S})})) & \text{otherwise}
\end{cases}
```

where:
- $\theta^*(\mathcal{S})$: Model parameters after applying bit flips in $\mathcal{S}$
- $\lambda = 5.0$: Penalty coefficient for accuracy violations

### 7. Attack Objective

The optimization problem:

```latex
\begin{aligned}
\mathcal{S}^* = \arg\max_{\mathcal{S} \subseteq \mathcal{C}} \quad & \text{ASR}(\mathcal{M}_{\theta^*(\mathcal{S})}, y_t) \\
\text{subject to} \quad & |\mathcal{S}| \leq B \\
& \text{Acc}(\mathcal{M}_{\theta^*(\mathcal{S})}) \geq \tau_{\text{acc}}
\end{aligned}
```

---

## Algorithm Pseudocode

### Algorithm 1: Bit Candidate Selection

```latex
\begin{algorithm}[H]
\caption{Select Bit Candidates}
\begin{algorithmic}[1]
\REQUIRE Model $\mathcal{M}_\theta$, dataset $\mathcal{D}$, layer $\ell$, candidates count $K$, $\alpha$
\ENSURE Candidate set $\mathcal{C}$
\STATE Sample batch $(\mathbf{X}, \mathbf{Y}) \sim \mathcal{D}$
\STATE Compute sensitivity: $\mathbf{S}^{(\ell)} \leftarrow \text{ComputeSensitivity}(\mathcal{M}_\theta, \ell, \mathbf{X}, \mathbf{Y}, \alpha)$
\STATE $\mathcal{I} \leftarrow \text{TopK}(\text{Flatten}(\mathbf{S}^{(\ell)}), K)$ \COMMENT{Top-K parameter indices}
\STATE $d \leftarrow \text{BitWidth}(\mathbf{W}^{(\ell)})$ \COMMENT{32 for FP32, 16 for FP16, 8 for INT8}
\STATE $\mathcal{B} \leftarrow \text{GetRelevantBitPositions}(d)$ \COMMENT{Prioritize sign/exponent}
\STATE $\mathcal{C} \leftarrow \emptyset$
\FOR{$\text{idx} \in \mathcal{I}$}
    \STATE $(i, j) \leftarrow \text{Unravel}(\text{idx}, \text{shape}(\mathbf{W}^{(\ell)}))$
    \FOR{$b \in \mathcal{B}$}
        \STATE $\mathcal{C} \leftarrow \mathcal{C} \cup \{(\ell, i, j, b)\}$
    \ENDFOR
\ENDFOR
\RETURN $\mathcal{C}$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 2: Compute Sensitivity

```latex
\begin{algorithm}[H]
\caption{Compute Hybrid Sensitivity}
\begin{algorithmic}[1]
\REQUIRE Model $\mathcal{M}_\theta$, layer $\ell$, inputs $\mathbf{X}$, targets $\mathbf{Y}$, $\alpha$
\ENSURE Sensitivity map $\mathbf{S}^{(\ell)}$
\STATE $\mathbf{W}_{\text{norm}}^{(\ell)} \leftarrow \|\mathbf{W}^{(\ell)}\|_2$ \COMMENT{L2 norm of weights}
\IF{$\alpha > 0$}
    \STATE $\mathcal{M}_\theta.\text{zero\_grad}()$
    \STATE $\hat{\mathbf{Y}} \leftarrow \mathcal{M}_\theta(\mathbf{X})$
    \STATE $\mathcal{L} \leftarrow \text{CrossEntropy}(\hat{\mathbf{Y}}, \mathbf{Y})$
    \STATE $\mathcal{L}.\text{backward}()$
    \STATE $\mathbf{G}_{\text{norm}}^{(\ell)} \leftarrow \|\nabla_{\mathbf{W}^{(\ell)}} \mathcal{L}\|_2$ \COMMENT{L2 norm of gradients}
    \STATE $\mathbf{S}^{(\ell)} \leftarrow \alpha \cdot \mathbf{G}_{\text{norm}}^{(\ell)} + (1-\alpha) \cdot \mathbf{W}_{\text{norm}}^{(\ell)}$
\ELSE
    \STATE $\mathbf{S}^{(\ell)} \leftarrow \mathbf{W}_{\text{norm}}^{(\ell)}$
\ENDIF
\RETURN $\mathbf{S}^{(\ell)}$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 3: Rank Layers by Sensitivity

```latex
\begin{algorithm}[H]
\caption{Rank Layers by Vulnerability}
\begin{algorithmic}[1]
\REQUIRE Model $\mathcal{M}_\theta$, dataset $\mathcal{D}$, layer info $\{\mathbf{W}^{(1)}, \ldots, \mathbf{W}^{(L)}\}$, $\alpha$
\ENSURE Ranked layers $[\ell_1, \ell_2, \ldots, \ell_L]$ by vulnerability
\STATE Sample batch $(\mathbf{X}, \mathbf{Y}) \sim \mathcal{D}$
\STATE $\mathcal{V} \leftarrow []$ \COMMENT{Vulnerability scores}
\FOR{$\ell = 1$ \TO $L$}
    \STATE $\mathbf{S}^{(\ell)} \leftarrow \text{ComputeSensitivity}(\mathcal{M}_\theta, \ell, \mathbf{X}, \mathbf{Y}, \alpha)$
    \STATE $k \leftarrow \max(1, \lfloor 0.001 \times |\mathbf{W}^{(\ell)}| \rfloor)$ \COMMENT{0.1\% of parameters}
    \STATE $\mathcal{I}_{\text{top}} \leftarrow \text{TopK}(\text{Flatten}(\mathbf{S}^{(\ell)}), k)$
    \STATE $\mathbf{W}^{(\ell)}_{\text{orig}} \leftarrow \mathbf{W}^{(\ell)}.\text{clone}()$ \COMMENT{Backup weights}
    \FOR{$\text{idx} \in \mathcal{I}_{\text{top}}$}
        \STATE $(i, j) \leftarrow \text{Unravel}(\text{idx}, \text{shape}(\mathbf{W}^{(\ell)}))$
        \STATE $w_{ij}^{(\ell)} \leftarrow -w_{ij}^{(\ell)}$ \COMMENT{Simulate attack}
    \ENDFOR
    \STATE $\mathcal{L}_{\text{after}} \leftarrow \text{Evaluate}(\mathcal{M}_\theta, \mathbf{X}, \mathbf{Y})$
    \STATE $\mathbf{W}^{(\ell)} \leftarrow \mathbf{W}^{(\ell)}_{\text{orig}}$ \COMMENT{Restore weights}
    \STATE $\mathcal{V}[\ell] \leftarrow \mathcal{L}_{\text{after}}$
\ENDFOR
\STATE $\text{Ranking} \leftarrow \text{ArgSort}(\mathcal{V}, \text{descending})$
\RETURN $\text{Ranking}$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 4: Bit Flip Operation

```latex
\begin{algorithm}[H]
\caption{Flip Bit in Parameter}
\begin{algorithmic}[1]
\REQUIRE Layer $\ell$, parameter index $(i,j)$, bit position $b$
\ENSURE Modified parameter $w_{ij}^{(\ell)*}$, original value $w_{ij}^{(\ell)}$
\STATE $w_{\text{old}} \leftarrow w_{ij}^{(\ell)}$
\IF{$\text{isnan}(w_{\text{old}})$ \OR $\text{isinf}(w_{\text{old}})$}
    \RETURN $(w_{\text{old}}, w_{\text{old}})$ \COMMENT{Skip invalid values}
\ENDIF
\STATE $\text{dtype} \leftarrow \text{GetDataType}(w_{ij}^{(\ell)})$
\IF{$\text{dtype} = \text{FP32}$}
    \STATE $u \leftarrow \text{FloatToUInt32}(w_{\text{old}})$ \COMMENT{Reinterpret as uint32}
    \STATE $u \leftarrow u \oplus 2^b$ \COMMENT{XOR to flip bit}
    \STATE $w_{\text{new}} \leftarrow \text{UInt32ToFloat}(u)$
\ELSIF{$\text{dtype} = \text{FP16}$}
    \STATE $u \leftarrow \text{FloatToUInt16}(w_{\text{old}})$
    \STATE $u \leftarrow u \oplus 2^b$
    \STATE $w_{\text{new}} \leftarrow \text{UInt16ToFloat}(u)$
\ELSIF{$\text{dtype} = \text{INT8}$}
    \STATE $w_{\text{new}} \leftarrow w_{\text{old}} \oplus 2^b$
\ENDIF
\STATE $w_{ij}^{(\ell)} \leftarrow w_{\text{new}}$ \COMMENT{Update model parameter}
\RETURN $(w_{\text{old}}, w_{\text{new}})$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 5: Genetic Optimization

```latex
\begin{algorithm}[H]
\caption{Genetic Algorithm for Bit Flip Optimization}
\begin{algorithmic}[1]
\REQUIRE Model $\mathcal{M}_\theta$, dataset $\mathcal{D}$, candidates $\mathcal{C}$, budget $B$, pop size $P$, generations $G$, target class $y_t$, threshold $\tau_{\text{acc}}$
\ENSURE Optimal bit flip set $\mathcal{S}^*$
\STATE $\mathcal{P}_0 \leftarrow \text{InitializePopulation}(\mathcal{C}, B, P)$ \COMMENT{Diverse initialization}
\STATE $\mathcal{S}^* \leftarrow \emptyset$, $F^* \leftarrow -\infty$
\STATE $\theta_{\text{orig}} \leftarrow \theta$ \COMMENT{Backup original weights}
\FOR{$g = 1$ \TO $G$}
    \STATE $\mathcal{F} \leftarrow []$ \COMMENT{Fitness scores}
    \FOR{$\mathcal{S}_i \in \mathcal{P}_{g-1}$}
        \STATE $\theta \leftarrow \theta_{\text{orig}}$ \COMMENT{Restore weights}
        \FOR{$(\ell, i, j, b) \in \mathcal{S}_i$}
            \STATE $\text{FlipBit}(\ell, i, j, b)$ \COMMENT{Apply bit flips}
        \ENDFOR
        \STATE $\text{Acc}_i \leftarrow \text{Accuracy}(\mathcal{M}_\theta, \mathcal{D})$
        \STATE $\text{ASR}_i \leftarrow \text{ASR}(\mathcal{M}_\theta, y_t, \mathcal{D})$
        \IF{$\text{Acc}_i \geq \tau_{\text{acc}}$}
            \STATE $\mathcal{F}[i] \leftarrow \text{ASR}_i$
        \ELSE
            \STATE $\mathcal{F}[i] \leftarrow \text{ASR}_i - 5.0 \cdot (\tau_{\text{acc}} - \text{Acc}_i)$
        \ENDIF
        \IF{$\mathcal{F}[i] > F^*$}
            \STATE $F^* \leftarrow \mathcal{F}[i]$, $\mathcal{S}^* \leftarrow \mathcal{S}_i$
        \ENDIF
    \ENDFOR
    \IF{$\text{ASR}(\mathcal{S}^*) > 0.75$ \OR \text{NoImprovement}$(g, 4)$}
        \STATE \textbf{break} \COMMENT{Early stopping}
    \ENDIF
    \STATE $\mathcal{P}_g \leftarrow \text{CreateNextGeneration}(\mathcal{P}_{g-1}, \mathcal{F}, P, B)$
\ENDFOR
\RETURN $\mathcal{S}^*$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 6: Create Next Generation

```latex
\begin{algorithm}[H]
\caption{Create Next Generation (GA)}
\begin{algorithmic}[1]
\REQUIRE Population $\mathcal{P}$, fitness scores $\mathcal{F}$, pop size $P$, budget $B$
\ENSURE Next generation $\mathcal{P}'$
\STATE $i^* \leftarrow \arg\max_i \mathcal{F}[i]$ \COMMENT{Best individual}
\STATE $\mathcal{P}' \leftarrow \{\mathcal{P}[i^*]\}$ \COMMENT{Elitism: keep best}
\WHILE{$|\mathcal{P}'| < P$}
    \STATE $\mathcal{S}_1 \leftarrow \text{TournamentSelection}(\mathcal{P}, \mathcal{F}, k=3)$
    \STATE $\mathcal{S}_2 \leftarrow \text{TournamentSelection}(\mathcal{P}, \mathcal{F}, k=3)$
    \STATE $\mathcal{S}_{\text{child}} \leftarrow \text{Crossover}(\mathcal{S}_1, \mathcal{S}_2)$
    \STATE $\mathcal{S}_{\text{child}} \leftarrow \text{Mutate}(\mathcal{S}_{\text{child}}, \mathcal{C}, B, p_{\text{mut}}=0.2)$
    \STATE $\mathcal{P}' \leftarrow \mathcal{P}' \cup \{\mathcal{S}_{\text{child}}\}$
\ENDWHILE
\RETURN $\mathcal{P}'$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 7: Tournament Selection

```latex
\begin{algorithm}[H]
\caption{Tournament Selection}
\begin{algorithmic}[1]
\REQUIRE Population $\mathcal{P}$, fitness scores $\mathcal{F}$, tournament size $k$
\ENSURE Selected individual $\mathcal{S}$
\STATE $\mathcal{T} \leftarrow \text{RandomSample}(\{1, \ldots, |\mathcal{P}|\}, k)$ \COMMENT{Random indices}
\STATE $i^* \leftarrow \arg\max_{i \in \mathcal{T}} \mathcal{F}[i]$ \COMMENT{Best in tournament}
\RETURN $\mathcal{P}[i^*]$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 8: Crossover and Mutation

```latex
\begin{algorithm}[H]
\caption{Crossover}
\begin{algorithmic}[1]
\REQUIRE Parents $\mathcal{S}_1, \mathcal{S}_2$
\ENSURE Child $\mathcal{S}_{\text{child}}$
\STATE $p \leftarrow \text{Random}(0, \min(|\mathcal{S}_1|, |\mathcal{S}_2|))$ \COMMENT{Crossover point}
\STATE $\mathcal{S}_{\text{child}} \leftarrow \text{Sort}(\text{Unique}(\mathcal{S}_1[:p] \cup \mathcal{S}_2[p:]))$
\RETURN $\mathcal{S}_{\text{child}}$
\end{algorithmic}
\end{algorithm}
```

```latex
\begin{algorithm}[H]
\caption{Mutation}
\begin{algorithmic}[1]
\REQUIRE Individual $\mathcal{S}$, candidates $\mathcal{C}$, budget $B$, mutation rate $p_{\text{mut}}$
\ENSURE Mutated individual $\mathcal{S}'$
\STATE $\mathcal{S}' \leftarrow \mathcal{S}$
\IF{$\text{Random}() < p_{\text{mut}}$}
    \IF{$|\mathcal{S}'| > 1$ \AND $\text{Random}() < 0.5$}
        \STATE Remove random element from $\mathcal{S}'$ \COMMENT{Deletion}
    \ELSIF{$|\mathcal{S}'| < B$}
        \STATE $c \leftarrow \text{RandomChoice}(\mathcal{C} \setminus \mathcal{S}')$ \COMMENT{Addition}
        \STATE $\mathcal{S}' \leftarrow \text{Sort}(\mathcal{S}' \cup \{c\})$
    \ENDIF
\ENDIF
\RETURN $\mathcal{S}'$
\end{algorithmic}
\end{algorithm}
```

### Algorithm 9: Complete Bit Flip Attack Pipeline

```latex
\begin{algorithm}[H]
\caption{Complete Bit Flip Attack}
\begin{algorithmic}[1]
\REQUIRE Model $\mathcal{M}_\theta$, dataset $\mathcal{D}$, target class $y_t$, budget $B$, threshold $\tau_{\text{acc}}$
\ENSURE Attacked model $\mathcal{M}_{\theta^*}$, attack results
\STATE $\text{Acc}_0, \text{ASR}_0 \leftarrow \text{Evaluate}(\mathcal{M}_\theta, \mathcal{D}, y_t)$ \COMMENT{Baseline}
\STATE $\theta_{\text{orig}} \leftarrow \theta$ \COMMENT{Backup original model}
\\
\STATE \COMMENT{\textbf{Phase 1: Layer Sensitivity Analysis}}
\STATE $\text{Ranking} \leftarrow \text{RankLayersBySensitivity}(\mathcal{M}_\theta, \mathcal{D}, \alpha=0.5)$
\STATE $\ell^* \leftarrow \text{Ranking}[0]$ \COMMENT{Most vulnerable layer}
\\
\STATE \COMMENT{\textbf{Phase 2: Bit Candidate Selection}}
\STATE $\mathcal{C} \leftarrow \text{SelectBitCandidates}(\mathcal{M}_\theta, \mathcal{D}, \ell^*, K=1000, \alpha=0.5)$
\\
\STATE \COMMENT{\textbf{Phase 3: Genetic Optimization}}
\STATE $\mathcal{S}^* \leftarrow \text{GeneticOptimization}(\mathcal{M}_\theta, \mathcal{D}, \mathcal{C}, B, P=50, G=20, y_t, \tau_{\text{acc}})$
\\
\STATE \COMMENT{\textbf{Phase 4: Apply Attack}}
\STATE $\theta \leftarrow \theta_{\text{orig}}$ \COMMENT{Restore to original}
\FOR{$(\ell, i, j, b) \in \mathcal{S}^*$}
    \STATE $(w_{\text{old}}, w_{\text{new}}) \leftarrow \text{FlipBit}(\ell, i, j, b)$
    \STATE Record: $(\ell, (i,j), b, w_{\text{old}}, w_{\text{new}})$
\ENDFOR
\\
\STATE \COMMENT{\textbf{Phase 5: Final Evaluation}}
\STATE $\text{Acc}_f, \text{ASR}_f \leftarrow \text{Evaluate}(\mathcal{M}_{\theta^*}, \mathcal{D}, y_t)$
\STATE $\Delta_{\text{Acc}} \leftarrow \text{Acc}_0 - \text{Acc}_f$
\STATE $\Delta_{\text{ASR}} \leftarrow \text{ASR}_f - \text{ASR}_0$
\\
\STATE \textbf{return} $\mathcal{M}_{\theta^*}$, $\{\text{Acc}_0, \text{Acc}_f, \Delta_{\text{Acc}}, \text{ASR}_0, \text{ASR}_f, \Delta_{\text{ASR}}, \mathcal{S}^*, |\mathcal{S}^*|\}$
\end{algorithmic}
\end{algorithm}
```

---

## Complexity Analysis

### Time Complexity

**Layer Sensitivity Analysis:**
```latex
\mathcal{O}(L \cdot N \cdot C_{\text{forward}})
```
where $L$ is number of layers, $N$ is dataset size, $C_{\text{forward}}$ is forward pass cost.

**Bit Candidate Selection:**
```latex
\mathcal{O}(K \cdot \log K + K \cdot d)
```
where $K$ is number of candidates, $d$ is bit width (typically 32).

**Genetic Optimization:**
```latex
\mathcal{O}(G \cdot P \cdot (B \cdot C_{\text{flip}} + N \cdot C_{\text{forward}}))
```
where $G$ is generations (≤20), $P$ is population size (50), $B$ is bit budget (≤100), $C_{\text{flip}}$ is flip cost.

**Total Attack Complexity:**
```latex
\mathcal{O}(L \cdot N \cdot C_{\text{forward}} + G \cdot P \cdot N \cdot C_{\text{forward}})
```

Dominated by genetic optimization phase, typically **10-20 minutes** for BERT-base.

### Space Complexity

```latex
\mathcal{O}(|\theta| + P \cdot B + K)
```
where:
- $|\theta|$: Model parameter count (stored in GPU memory)
- $P \cdot B$: Population storage
- $K$: Candidate storage

For BERT-base: $|\theta| \approx 110M$, $P \cdot B = 50 \times 100 = 5K$, $K = 1000$ → dominated by model size.

---

## Usage Example in LaTeX Paper

Here's how to include these in a paper:

```latex
\section{Methodology}

\subsection{Hybrid Sensitivity Metric}

We define a hybrid sensitivity metric that combines weight magnitude and gradient information:

\begin{equation}
S(w_{ij}^{(\ell)}) = \alpha \cdot \tilde{g}_{ij}^{(\ell)} + (1-\alpha) \cdot \tilde{m}_{ij}^{(\ell)}
\end{equation}

where $\tilde{m}_{ij}^{(\ell)}$ and $\tilde{g}_{ij}^{(\ell)}$ are min-max normalized weight magnitudes and gradients respectively, and $\alpha \in [0,1]$ balances the two components.

\subsection{Attack Algorithm}

Our attack pipeline is formalized in Algorithm~\ref{alg:complete_attack}. The key insight is to leverage genetic optimization to efficiently search the combinatorially large space of possible bit flip combinations while maintaining model functionality.

% Include Algorithm 9 here
```

---

## Key Takeaways

1. **Hybrid Sensitivity**: Balancing gradient and magnitude provides robust vulnerability assessment across different model architectures

2. **Bit Position Priority**: Sign and exponent bits have exponentially higher impact than mantissa bits in floating-point representations

3. **Genetic Optimization**: Efficiently explores $\binom{K}{B}$ combinations (e.g., $\binom{1000}{10} \approx 2.6 \times 10^{23}$)

4. **Fitness Design**: The penalty term ($\lambda = 5.0$) ensures attacks remain stealthy by maintaining model functionality

5. **Attack Efficiency**: Flipping $B \ll |\theta|$ bits (typically $B \approx 5-10$, $|\theta| \approx 10^8$) achieves significant attack success
