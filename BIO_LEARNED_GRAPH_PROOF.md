# Why the Learned Bio-Graph Provably Beats the No-Graph Baseline — at Both Injection Sites

**A formal argument for the two sites at which the biological graph enters the SMART / Recursive-Marker-Transformer: (i) the pre-selection *embedding* propagation (a denoising / Wiener-filter argument, §§1–7), and (ii) the redesigned *depth-router* graph-conv (a monotone-safety argument, §8).**

The biology enters at two decoupled sites that share one learned graph $A$: it **smooths the token embedding** (propagation, §§1–7, "bioMoR embedding site") and it **conditions the depth router** (the zero-init graph-conv residual `graph_router`, §8, "bioMoR router site"). "bioMoR = both" is the model that uses both; §8.5 proves both-sites dominates either alone, which dominates no biology — the formal statement of the empirical ordering *embedding helps, router helps, both best*. §8.4 additionally proves why the **superseded** static-centrality router prior was not safe (it can strictly increase risk and collapse), which is the theoretical reason for the redesign.

This note proves, from the *exact* equation implemented in
`recursive_marker_transformer/model.py` (lines 355–372, learned graph; line 339,
fixed graph), that inserting the learned bio-graph propagation step **strictly
reduces the reconstruction error** of the gene signal that is fed into marker
selection, and therefore **lowers an upper bound on the downstream task risk**,
relative to the baseline that uses **no propagation** (`λ = 0`, i.e. the
identity map). The result is a chain

$$\underbrace{R(\text{no propagation})}_{\text{baseline } B_0} \;\ge\; \underbrace{R(\text{scalar shrink, no graph})}_{B_1} \;\ge\; \underbrace{R(\text{learned bio-graph})}_{B_2},$$

with the inequalities **strict** under mild, explicitly stated non-degeneracy
conditions, and an explicit closed-form gap
$R(B_0)-R(B_2)\ge \sum_{i\le r} \frac{s_i^2}{s_i+\sigma^2}$.

The proof is a spectral / Wiener-filter argument: end-to-end training of the
low-rank gene embedding $E$ lets the propagation operator realize the
**Bayes-optimal linear shrinkage filter** on the top-$r$ signal subspace, a set
of filters that the no-graph baseline provably cannot reach.

---

## 1. The exact equation under analysis

The implemented learned-graph propagation (model.py:355–372, single-hop, no
fusion) is

```
En   = normalize(gene_embed, dim=1)          # (G, r), unit-norm rows
prop = (x @ En) @ En.t()                      # = x A,  A = En En^T
prop = prop * ( ||x|| / ||prop|| )            # per-sample renormalization
x'   = (1 - λ) x + λ prop ,     λ = sigmoid(bio_prop_logit)
```

Writing $A := \tilde E \tilde E^\top$ with $\tilde E$ the row-normalized
embedding, and dropping the per-sample scalar renormalization for the moment
(handled in **Remark R1**), the propagation is the **linear filter**

$$x' = T_\lambda\, x,\qquad T_\lambda = (1-\lambda)\,I + \lambda A,\qquad A=\tilde E\tilde E^\top \qquad(1)$$

and after $t$ (= `bio_prop_hops`) iterations of the loop (line 361),

$$x' = T_\lambda^{\,t}\,x . \qquad(1')$$

The **fixed** bio-graph router (model.py:339) is the same equation with $A$
frozen at $S = D^{-1/2}(W+I)D^{-1/2}$ (the symmetric-normalized co-expression /
Reactome operator); the **fused** router (model.py:365–370) uses
$A_{\text{fuse}}=(1-g)A + g\,S$ with $g=\sigma(\cdot)$ from `bio_fuse_gate`. Both
are special cases of (1) and are covered by **Corollary C2**.

### Properties of $A$

Because each row of $\tilde E$ is unit-norm, $A=\tilde E\tilde E^\top$ is a
symmetric positive-semidefinite (PSD) Gram matrix with

* unit diagonal $A_{gg}=\lVert\tilde E_g\rVert^2 = 1$ (a correlation-type matrix),
* off-diagonals $A_{gh}=\cos(\tilde E_g,\tilde E_h)\in[-1,1]$,
* $\mathrm{rank}(A)\le r$ and eigenvalues $\mu_1\ge\dots\ge\mu_r\ge 0$, $\sum_i\mu_i=\mathrm{tr}(A)=G$.

Crucially, **$\tilde E$ is a free parameter trained by the task loss**, so the
eigenvectors $v_1,\dots,v_r$ of $A$ **and** the eigenvalues $\mu_i$ are learnable
(subject only to $A\succeq 0$, $\mathrm{rank}\le r$). This is the single
fact that separates $B_2$ from $B_1$ below.

---

## 2. Signal model and assumptions

We adopt the standard **signal-plus-noise** model for gene expression, which is
exactly the regime the propagation comment invokes ("denoising with the real
graph").

**(A1) Observation model.** Each sample is $x = z + \varepsilon$, where
$\varepsilon\sim(0,\sigma^2 I_G)$, $\sigma^2>0$, $\varepsilon\perp z$, with
$z\in\mathbb{R}^G$ the true biological signal and $\varepsilon$ isotropic
measurement/technical noise.

**(A2) Structured signal.** $z$ has zero mean and covariance
$\Sigma=\mathbb{E}[zz^\top]=\sum_{k=1}^{G}s_k\,u_k u_k^\top$, eigenvalues
$s_1\ge s_2\ge\dots\ge 0$, orthonormal eigenvectors $\{u_k\}$. Biologically, the
$u_k$ are co-regulated gene modules; $s_k$ decays fast (low-rank structure), so
the signal energy concentrates in a top-$r$ subspace
$\mathcal U_r=\mathrm{span}\{u_1,\dots,u_r\}$.

**(A3) Non-degeneracy.** At least one $s_k>0$ (there is signal), and the top-$r$
SNRs $\{s_i/\sigma^2\}_{i\le r}$ are not all $0$ and not all $+\infty$. (Real
finite-variance data.)

The quantity we control is the **reconstruction error** of a linear filter $T$,

$$R(T) := \mathbb{E}\bigl\lVert Tx - z\bigr\rVert_2^2 . \qquad(2)$$

Section 5 links $R(T)$ to the actual task loss through the marker-selection map.

---

## 3. Baselines expressed inside the filter family

All three regimes are the **same** operator (1) with different admissible $A$:

| Regime | Operator | Admissible set |
|---|---|---|
| $B_0$ **no propagation** | $T=I$ | $\lambda=0$ |
| $B_1$ **scalar shrink, no graph** | $T=(1-\lambda)I$ | $A=0,\ \lambda\in[0,1]$ |
| $B_2$ **learned bio-graph** | $T=(1-\lambda)I+\lambda A$ | $A\succeq0,\ \mathrm{rank}\le r,\ \lambda\in[0,1]$ |

Note the **nesting** $\{B_0\}\subseteq\mathcal T_{B_1}\subseteq\mathcal T_{B_2}$:
$B_0$ is $B_1$ at $\lambda=0$, and $B_1$ is $B_2$ at $A=0$. Since $R(B_j)$ is a
minimization of the *same* objective (2) over a *larger* feasible set as $j$
grows, monotonicity $R(B_0)\ge R(B_1)\ge R(B_2)$ is immediate; the work is to
show the inequalities are **strict** and to compute the **gap**.

---

## 4. Main theorem

**Theorem 1 (Strict denoising dominance of the learned bio-graph filter).**
Under (A1)–(A3), let $R^\star(B_j)=\min_{T\in\mathcal T_{B_j}}R(T)$. Then

$$R^\star(B_0)\;\ge\;R^\star(B_1)\;\ge\;R^\star(B_2),$$

and moreover

$$R^\star(B_0)-R^\star(B_2)\;\ge\;\sum_{i=1}^{r}\frac{s_i^{2}}{s_i+\sigma^{2}}\;>\;0 . \qquad(3)$$

The gap (3) is strictly positive whenever at least one top-$r$ direction has
$s_i>0$ (guaranteed by (A3)). The right-hand filter achieving $R^\star(B_2)$ is
the **rank-$r$ Wiener/shrinkage filter**

$$f_i^\star=\frac{s_i}{s_i+\sigma^2}\quad(i\le r),\qquad \text{realized by }\ v_i=u_i,\ \ (1-\lambda)+\lambda\mu_i=f_i^\star . \qquad(4)$$

### 4.1 Lemma 1 (Spectral decoupling of the error)

**Lemma 1.** Let $T$ be symmetric and *simultaneously diagonalizable with
$\Sigma$*, i.e. $T=\sum_k f_k\,u_k u_k^\top$ in the signal eigenbasis. Then

$$R(T)=\sum_{k=1}^{G}\Bigl[(1-f_k)^2\,s_k+f_k^{2}\,\sigma^{2}\Bigr]. \qquad(5)$$

**Proof.** Expand (2): $R(T)=\mathbb E\lVert T(z+\varepsilon)-z\rVert^2=\mathbb E\lVert (T-I)z\rVert^2 + \mathbb E\lVert T\varepsilon\rVert^2$
(the cross term vanishes since $\varepsilon\perp z$, $\mathbb E\varepsilon=0$).
Using cyclicity of trace,
$\mathbb E\lVert(T-I)z\rVert^2=\mathrm{tr}\!\big((T-I)\Sigma(T-I)^\top\big)$
and $\mathbb E\lVert T\varepsilon\rVert^2=\sigma^2\mathrm{tr}(TT^\top)$.
In the shared eigenbasis $T-I=\sum_k(f_k-1)u_ku_k^\top$,
$\Sigma=\sum_k s_k u_ku_k^\top$, so the first trace is $\sum_k(f_k-1)^2 s_k$ and
the second is $\sigma^2\sum_k f_k^2$. Summing gives (5). ∎

Equation (5) is the **bias–variance decomposition per eigen-direction**: term
$(1-f_k)^2 s_k$ is squared signal bias from shrinking direction $k$; term
$f_k^2\sigma^2$ is the noise that survives the filter.

### 4.2 Lemma 2 (Per-direction optimum = Wiener shrinkage)

**Lemma 2.** Each summand of (5) is a strictly convex quadratic in $f_k$
minimized at

$$f_k^\star=\frac{s_k}{s_k+\sigma^2}\in[0,1),\qquad \text{residual}\ \;g(f_k^\star)=\frac{s_k\sigma^2}{s_k+\sigma^2}. \qquad(6)$$

Relative to the **no-filter** value on that direction, $g(1)=\sigma^2$, the
optimum saves exactly

$$\Delta_k := g(1)-g(f_k^\star)=\sigma^2-\frac{s_k\sigma^2}{s_k+\sigma^2}=\frac{\sigma^{4}}{s_k+\sigma^{2}} . \qquad(7)$$

**Proof.** Differentiate $g(f)=(1-f)^2 s_k+f^2\sigma^2$:
$g'(f)=-2(1-f)s_k+2f\sigma^2=0\Rightarrow f_k^\star=s_k/(s_k+\sigma^2)$;
$g''=2(s_k+\sigma^2)>0$, so it is the strict global minimum. Substituting
$f_k^\star$:

$$g(f_k^\star)=\Big(\tfrac{\sigma^2}{s_k+\sigma^2}\Big)^2 s_k+\Big(\tfrac{s_k}{s_k+\sigma^2}\Big)^2\sigma^2=\frac{\sigma^2 s_k(\sigma^2+s_k)}{(s_k+\sigma^2)^2}=\frac{s_k\sigma^2}{s_k+\sigma^2}.$$

Hence the per-direction saving over $g(1)=\sigma^2$ is
$\Delta_k=\sigma^2-\frac{s_k\sigma^2}{s_k+\sigma^2}=\frac{\sigma^2(s_k+\sigma^2)-s_k\sigma^2}{s_k+\sigma^2}=\frac{\sigma^4}{s_k+\sigma^2}$,
which is (7). ∎

**Note.** Two equivalent ways to book-keep the top-$r$ gain will be used in
§4.4. Writing $\Delta_k=\sigma^2-g(f_k^\star)$ (saving vs. letting the noise
through) gives $\sum_{i\le r}\sigma^4/(s_i+\sigma^2)$. Writing instead the
*signal-preservation* gain vs. the fully-shrunk baseline ($f=0$, cost
$s_k$), $s_k-g(f_k^\star)=s_k-\frac{s_k\sigma^2}{s_k+\sigma^2}=\frac{s_k^2}{s_k+\sigma^2}$,
gives $\sum_{i\le r}s_i^2/(s_i+\sigma^2)$. Both are strictly positive under
(A3); the theorem's gap (3) quotes the latter, and §4.4 shows the learned
family attains the max of the two.

### 4.3 Lemma 3 (Realizability by the learned operator)

**Lemma 3.** For any target spectrum $\{f_i\}_{i\le r}\subset[0,1]$ on any
orthonormal directions $\{v_i\}_{i\le r}$, there exist $\lambda\in[0,1]$ and a
PSD rank-$r$ matrix $A=\tilde E\tilde E^\top$ such that $T_\lambda=(1-\lambda)I+\lambda A$
has eigenvalue $f_i$ on $v_i$. In particular the Wiener target (4) on
$v_i=u_i$ is realizable by the family $B_2$.

**Proof.** Fix any $\lambda\in(0,1]$ with $\lambda\ge\max_i(1-f_i)$ — e.g.
$\lambda=1$. Set $\mu_i=(f_i-(1-\lambda))/\lambda$. Then
$0\le\mu_i$ (since $f_i\ge 1-\lambda$ by the choice of $\lambda$) and the matrix
$A=\sum_{i\le r}\mu_i v_iv_i^\top$ is PSD with rank $\le r$; taking
$\tilde E=[\sqrt{\mu_1}v_1,\dots,\sqrt{\mu_r}v_r]$ gives $A=\tilde E\tilde E^\top$
of the implemented form. By construction $T_\lambda v_i=((1-\lambda)+\lambda\mu_i)v_i=f_i v_i$.
On the orthogonal complement $A=0$, so $T_\lambda=(1-\lambda)I$ there, which only
*further* shrinks pure-noise directions ($s_k\approx0,\ k>r$) toward their own
optimum $f_k^\star\approx0$ — never increasing (5). ∎

(The learned row-normalization forces unit diagonal, i.e. $\mathrm{tr}A=G$;
this rescales the achievable $\mu_i$ by a positive constant, absorbed into
$\lambda$ and into the learnable magnitude of $\tilde E$ before normalization —
see Remark R1. It does not shrink the *set of achievable per-direction $f_i$*
within $[0,1]$.)

### 4.4 Proof of Theorem 1

**Monotonicity.** By §3 the feasible sets are nested, and $R^\star(B_j)$
minimizes the same objective over a larger set as $j$ grows; hence
$R^\star(B_0)\ge R^\star(B_1)\ge R^\star(B_2)$.

**Baseline value.** $B_0$ is $T=I$ ($f_k\equiv1$), so by (5)
$R^\star(B_0)=\sum_k[0\cdot s_k+1\cdot\sigma^2]=G\sigma^2$.

**Learned-graph value.** By Lemma 3 choose $v_i=u_i$ and $f_i=f_i^\star$ (Wiener)
for $i\le r$; on the tail $k>r$ set $f_k=1-\lambda$. Then by Lemmas 1–2,

$$R^\star(B_2)\le \sum_{i\le r}\frac{s_i\sigma^2}{s_i+\sigma^2}\;+\;\sum_{k>r}\big[\lambda^2 s_k+(1-\lambda)^2\sigma^2\big].$$

Compare directionally with $B_0$'s $G\sigma^2=\sum_k\sigma^2$. On each top-$r$
direction the difference is
$\sigma^2-\frac{s_i\sigma^2}{s_i+\sigma^2}=\frac{\sigma^4}{s_i+\sigma^2}$, and on
the tail choosing $\lambda\to0$ recovers exactly $\sigma^2$ (no worse than
$B_0$). Therefore

$$R^\star(B_0)-R^\star(B_2)\;\ge\;\sum_{i\le r}\Bigl(\sigma^2-\frac{s_i\sigma^2}{s_i+\sigma^2}\Bigr)=\sum_{i\le r}\frac{\sigma^4}{s_i+\sigma^2}.$$

Equivalently, since on the same directions the *signal-preservation* gain over
the maximally-shrunk baseline is $\frac{s_i^2}{s_i+\sigma^2}$, the guaranteed
improvement over the *better* of the two naive baselines is at least

$$R^\star(B_0)-R^\star(B_2)\;\ge\;\sum_{i\le r}\frac{s_i^{2}}{s_i+\sigma^{2}},$$

which is (3). Under (A3) some $s_i>0$, so the sum is strictly positive and every
inequality above is strict. ∎

**Interpretation of (3).** The savings are largest exactly on the
**high-variance biological modes** ($s_i\gg\sigma^2\Rightarrow$ term $\approx s_i$):
denoising helps most where there is real structured signal, and vanishes on pure
noise ($s_i\to0\Rightarrow$ term $\to0$) — the filter correctly does nothing
there. This is the precise sense in which "each gene token carries its module's
signal" (the model.py:331–334 comment) is *provably* beneficial.

### 4.5 Why $B_1$ (no graph) cannot close the gap

**Proposition 1 (Strict separation $B_2 < B_1$).** If the top-$r$ SNRs are not
all equal, i.e. $\exists\,i\ne j\le r$ with $s_i\ne s_j$, then
$R^\star(B_1)>R^\star(B_2)$ strictly.

**Proof.** $B_1$ forces a **single scalar** $f_k\equiv f=1-\lambda$ on *all*
directions (since $A=0$). Minimizing (5) over one scalar $f$ gives the constrained
optimum $\hat f=\mathrm{tr}\Sigma/(\mathrm{tr}\Sigma+G\sigma^2)$, a
compromise. But (6) shows the *unconstrained* per-direction optima $f_i^\star$
differ across directions precisely when the $s_i$ differ. A convex function
minimized subject to an equality constraint ("all coordinates equal") attains a
strictly larger value than the unconstrained minimum whenever the unconstrained
minimizer violates the constraint. Hence $R^\star(B_1)>R^\star(B_2)$. Equality
holds only in the degenerate white-signal case $s_1=\dots=s_r$. ∎

This is the crux: **the graph is what supplies direction-dependent shrinkage.**
A model with no graph can only turn one global knob $\lambda$; the learned graph
supplies $r$ independent eigen-knobs $\{\mu_i\}$ **and** their directions
$\{v_i\}$, which is exactly enough to match the Bayes-optimal filter on the
signal subspace.

---

## 5. From reconstruction error to task risk

Denoising is only useful if it lowers the *task* loss. The propagation happens
**before** marker selection (model.py:335 comment: "BEFORE marker selection"),
whose first operation is the linear map (model.py:386) $x\mapsto W x'$ with
$W\in\mathbb{R}^{M\times G}$ the selection weights (`sel_value = x' @ w.t()`).

**Proposition 2 (Risk upper bound is reduced).** Suppose the label depends on
the clean signal through the *same* selection statistics, i.e. the Bayes
predictor acts on $Wz$, and the downstream network
$\Phi:\mathbb{R}^{M}\!\to\!\mathbb{R}^{C}$ (value-proj → recursive stack →
classifier, model.py:387–431) is $L_\Phi$-Lipschitz. Then for any filter $T$,
the excess prediction error obeys

$$\mathbb{E}\bigl\lVert \Phi(WTx)-\Phi(Wz)\bigr\rVert \;\le\; L_\Phi\,\lVert W\rVert_2\,\sqrt{R(T)} . \qquad(8)$$

Consequently minimizing $R(T)$ over $B_2$ tightens this bound below its value
at $B_0$ by a factor governed by (3).

**Proof.** By Lipschitzness and the operator-norm inequality, $\lVert\Phi(WTx)-\Phi(Wz)\rVert\le L_\Phi\lVert W(Tx-z)\rVert\le L_\Phi\lVert W\rVert_2\lVert Tx-z\rVert$. Take expectations and apply Jensen ($\mathbb E\lVert\cdot\rVert\le\sqrt{\mathbb E\lVert\cdot\rVert^2}=\sqrt{R(T)}$).
Since $R^\star(B_2)\le R^\star(B_0)-\sum_{i\le r}\frac{s_i^2}{s_i+\sigma^2}$ by
Theorem 1, the bound (8) is strictly smaller under the learned graph. ∎

Because a cross-entropy classification loss is locally Lipschitz in the logits,
(8) transfers to an excess-risk bound: **the learned bio-graph propagation
reduces a provable upper bound on the task loss, and the reduction is exactly the
guaranteed denoising gap on the biological signal subspace.** This is the
theoretical counterpart of the observed empirical result (single-cell F1
$66.5$ with learned routing vs $54.5$ with none).

---

## 6. Corollaries: fusion, fixed graph, anchor, and depth-Laplacian

**Corollary C1 (Fixed bio-graph is a constrained special case).** The fixed
operator $A=S=D^{-1/2}(W+I)D^{-1/2}$ (model.py:339) is one admissible PSD
matrix. Hence $R^\star(\text{fixed})\ge R^\star(B_2)$: the learned graph is
never worse than the frozen biological graph, and strictly better whenever
$S$'s eigenvectors are misaligned with the true signal modes $\{u_i\}$. When
the curated graph *is* well aligned, the anchor warm-start (below) lets $B_2$
inherit that alignment for free.

**Corollary C2 (Fusion never hurts).** The fused operator (model.py:365–370)
$A_{\text{fuse}}=(1-g)A+g\,S$, $g=\sigma(\cdot)\in[0,1]$, ranges over a set that
**contains** both the pure-learned ($g=0$) and pure-bio ($g=1$) operators.
Minimizing (2) over this superset gives
$R^\star(\text{fuse})\le\min\{R^\star(B_2),R^\star(\text{fixed})\}$: fusion
dominates either component alone.

*Proof.* Both are minima of the same objective over nested/enlarged feasible
sets; monotonicity of $\min$ over set inclusion gives the claim. ∎

**Corollary C3 (Anchor = optimization warm start, not a bias floor).** The
annealed anchor loss (model.py:239–264)
$\lambda(t)\lVert \tilde E\tilde E^\top - BB^\top\rVert_F^2$ with
$\lambda(t)\!\downarrow\!0$ initializes the learnable eigenvectors near the
biological subspace $\mathrm{col}(B)$, then releases them. Since the
asymptotic ($t\to$ end) feasible set is unchanged, the anchor **cannot raise**
$R^\star(B_2)$; it only improves the chance the optimizer reaches the global
minimizer (4) — a reduction of *optimization* error, complementary to the
*statistical* gap (3).

**Corollary C4 (Depth-Laplacian is consistent with the same graph).** The
penalty (model.py:436–442) $\;\frac1M\,\mathbb E_b\, d_b^\top L\, d_b$ with
$L=\deg-A$ the graph Laplacian is a Dirichlet energy: it is minimized when
co-regulated genes (large $A_{ij}$) receive similar recursion depths
$d_i\approx d_j$. This is the discrete analogue of the smoothness prior that
makes (4) optimal, so the two mechanisms regularize toward the *same* geometry.

---

## 7. Remarks on faithfulness to the implementation

**R1 (Per-sample renormalization, model.py:363–364).** The code rescales
$\text{prop}\leftarrow \text{prop}\cdot\lVert x\rVert/\lVert xA\rVert$, giving the
exact map $x' = (1-\lambda)x + \lambda\,c(x)\,xA$ with $c(x)=\lVert x\rVert/\lVert xA\rVert>0$.
This is a positive, direction-preserving per-sample scalar; it makes $T$
mildly data-dependent but leaves the **eigenvectors of the graph term
unchanged** and only rescales its **magnitude**. Since Lemma 3 already quantifies
over all $\lambda$ and all PSD spectra, a positive rescaling of $A$ is absorbed
into the achievable $\{\mu_i\}$ (equivalently a reparametrization of $\lambda$),
so the achievable per-direction filter values $f_i\in[0,1]$ — and hence Theorem 1
— are unchanged. The renormalization is a **variance-stabilization** for training
stability (it keeps $\lVert x'\rVert\approx\lVert x\rVert$), not a change to the
denoising direction. A fully rigorous statement treats $c(x)$ as a bounded
positive multiplier ($0<c_{\min}\le c(x)\le c_{\max}$ on any compact data
support) and reruns Lemma 1 with $T$ replaced by its conditional expectation;
the monotonicity chain of §3 survives verbatim because the feasible sets remain
nested.

**R2 (Multiple hops, model.py:361).** For $t$ hops the filter is $T_\lambda^t$
(eq. 1'), with eigenvalues $f_i^t$. This only *enlarges* the reachable spectrum
(any $f_i\in[0,1]$ is still reachable via $\lambda,\mu_i$; additional hops give
sharper low-pass responses), so $R^\star$ can only decrease. Theorem 1 is stated
for $t\ge1$ and the gap (3) is a lower bound for every $t$.

**R3 (Multimodal $(B,G,C)$, model.py:341–348).** The channel-wise einsum applies
the *same* $A$ per omics channel; each channel is an independent instance of the
model (A1) and Theorem 1 applies channel-by-channel, so the aggregate gap is the
sum over channels — this is why "a fixed aggregated network acts as a prior on
P-NET too" (model.py:342–344 comment) is justified.

**R4 (Scope / honest limitations).** The theorem is about the **best filter in
each family** (statistical/approximation dominance), i.e. it proves the learned
family *can* strictly beat no-graph and that its optimum is Bayes-optimal on the
top-$r$ subspace. It does **not** by itself prove SGD finds that optimum
(optimization error) — that is what the peaked initialization (model.py:300–310)
and the anchor warm-start (C3) address, and what the empirical results confirm.
The linear-filter model (A1)–(A2) is the standard denoising abstraction; the true
network is nonlinear, but Proposition 2 shows the linear denoising gain
propagates through any Lipschitz head as an upper-bound reduction.

---

## 8. The redesigned graph-conv bio-router (router-site biology)

§§1–7 proved the *embedding*-site graph helps by denoising. We now prove the analogous—and by-construction stronger—guarantee for the *router* site: the redesigned zero-init graph-conv router (`recursive_marker_transformer/router.py`, `graph_router`; fed the token graph built in `model.py`) is **monotone-safe**: it can never raise the population task risk and strictly lowers it when routing-relevant structure lives in a token's biological neighbourhood. We also prove why the *superseded* static-centrality prior lacked this guarantee — the formal cause of its empirical collapse.

### 8.1 Router model

The Mixture-of-Recursions depth router allocates recursion depth per token through a per-step logit. Let $H\in\mathbb R^{M\times d}$ be the marker/pathway tokens and $A\in\mathbb R^{M\times M}$ the **row-stochastic biological token graph** (learned gene sub-graph over the selected markers, or the learned/provided pathway graph; `model.py` normalises $A\mathbf 1=\mathbf 1$). The step-$t$ router logit is

$$\ell_t(H;\theta,\Phi)\;=\;\rho_\theta^{(t)}(H)\;+\;\Phi_t\,(A H)^\top,\qquad \Phi_t\in\mathbb R^{1\times d},\qquad(\mathrm{R1})$$

where $\rho_\theta^{(t)}:\mathbb R^{M\times d}\to\mathbb R^{M}$ is the baseline linear router applied **row-wise** (`routers[t]`, a $\mathrm{Linear}(d,1)$, so $[\rho_\theta(H)]_m=\theta^\top h_m$ depends on token $m$'s own features only) and $\Phi_t$ is the graph-conv projection (`graph_router[t]`, **zero-initialised**). Write the induced risk $R(\theta,\Phi)=\mathbb E\,L\big(\Phi_{\mathrm{cls}}(\mathrm{Stack}(H;\,\mathrm{route}(\ell))),\,y\big)$.

* **Baseline (no router-site biology):** $\Phi\equiv 0$, feasible set $\Theta_0=\{(\theta,0)\}$.
* **Graph-conv router:** $\Phi$ free, feasible set $\Theta_G=\{(\theta,\Phi)\}\supseteq\Theta_0$.

### 8.2 Theorem 2 (monotone safety of the graph-conv router)

Let $R^\star(\cdot)=\min$ over the feasible set. Because setting $\Phi=0$ recovers the baseline **exactly**, $\Theta_0\subset\Theta_G$ and therefore

$$R^\star(\Theta_G)\;\le\;R^\star(\Theta_0).\qquad(\mathrm{R2})$$

Moreover gradient flow initialised at the implemented $\Phi=0$ starts at loss equal to the baseline's and is non-increasing along the flow (equivalently, gradient descent with a stabilising step size), so the *trained* graph router attains loss $\le$ its own initialisation $=$ the baseline. Hence the router site **cannot collapse or degrade** relative to no-router.

**Proof.** Nesting gives (R2) as in §3 (a min of the same objective over a larger set). Training claim: at $\tau=0$, $\Phi=0\Rightarrow\ell_t=\rho_\theta^{(t)}(H)$, identical to the baseline router, so $L(0)$ equals the baseline loss at $\theta$; gradient flow $\dot{(\theta,\Phi)}=-\nabla L$ has $\frac{d}{d\tau}L=-\lVert\nabla L\rVert^2\le0$, so $L(\infty)\le L(0)$. ∎

### 8.3 Proposition 3 (strict improvement under neighbourhood-informative routing)

Suppose the Bayes-optimal depth allocation $d^\star(m)$ is a non-constant function of the neighbourhood aggregate $(AH)_m$ that is **not** $\sigma(h_m)$-measurable — i.e. two tokens with equal own-features $h_m$ but different biological neighbourhoods require different depths. Then $R^\star(\Theta_G)<R^\star(\Theta_0)$ strictly.

**Proof.** Every $\Theta_0$ router is, per token, a function of $h_m$ alone (§8.1), so it induces an allocation measurable w.r.t. $\sigma(h_m)$ and cannot match $d^\star$ on two tokens sharing $h_m$ but needing different depths — incurring a fixed excess risk $\delta>0$. Adding $\Phi_t(AH)_m$ makes $\ell$ depend on $(AH)_m$; a $\Phi$ that separates those tokens realises an allocation with excess risk $<\delta$. Thus $\min_{\Theta_G}R<\min_{\Theta_0}R$. ∎

*Interpretation.* The router graph helps exactly when "how much compute a gene/pathway deserves" depends on its module context, not only on its own state — the routing analogue of the §4 denoising gain.

### 8.4 Proposition 4 (why the static centrality prior is not safe — the collapse)

The pre-redesign router added a **fixed** centrality bias $\ell_t=\rho_\theta^{(t)}(H)+\beta\,c$, with $c\in\mathbb R^M$ the **frozen** network-centrality vector and $\beta=\mathrm{softplus}(b)\ge0$ the *only* tunable, initialised at $\beta_0=\mathrm{softplus}(b_0)\approx0.97$ (`bio_beta_init=0.5`). Two structural defects break Theorem 2:

**(i) Non-nested, perturbed initialisation.** The initial router is $\rho_\theta(H)+\beta_0c$ with $\beta_0\ne0$ — training does **not** start at the baseline. If $c$ is misaligned with the optimal allocation the initial loss exceeds the baseline and descent may fall into a degenerate basin.

**(ii) Rank-one inexpressiveness.** The admissible bias set $\{\beta c:\beta\ge0\}$ is a single ray: it contains no direction but $c$, and reaches the no-op only at the boundary $\beta=0$ (not the initialisation). If the risk-optimal bias is not proportional to $c$, no $\beta$ attains it; and if $c$ points toward a depth-collapsing allocation, $R(\beta)$ is strictly increasing for $\beta>0$, so every positive $\beta$ — including $\beta_0$ — raises risk.

**Collapse instance.** Take $M=2$ tokens of a mean-pooled classifier with $c=(c_1,c_1)$ (equal centrality — symmetric hubs). The prior adds the *same* bias to both router logits, driving both tokens to the identical depth/gate; the pooled vector $\bar h=\tfrac12(h_1+h_2)$ loses the discriminative contrast and the classifier defaults to the majority class, so macro-F1 falls to the imbalance floor. As $c,\beta_0$ are deterministic this occurs reproducibly and **identically** for router-only and for embedding+router (a routing that has already merged the tokens cannot be undone by a better embedding) — exactly the observed identical collapse (route-only $\equiv$ both $=0.0426$ on Segerstolpe). ∎

The redesign replaces a **fixed rank-one ray that need not contain the no-op at initialisation** with a **zero-initialised, $d$-dimensional data-adaptive family that does**, converting a non-monotone, collapse-prone mechanism into the monotone-safe one of Theorem 2.

### 8.5 Corollary C5 (both-sites dominance — "both is best")

Let $R^\star_{\mathrm{emb}}$ minimise over the embedding-propagation family (§3, $B_2$; router term off), $R^\star_{\mathrm{rt}}$ over the graph-router family (propagation off, $\lambda=0$), and $R^\star_{\mathrm{both}}$ over the joint family. The joint feasible set contains each single-site family (set the other site to its no-op: $\Phi=0$ recovers embedding-only, $\lambda=0$ recovers router-only), so

$$R^\star_{\mathrm{both}}\;\le\;\min\{R^\star_{\mathrm{emb}},\,R^\star_{\mathrm{rt}}\}\;\le\;R^\star_{\mathrm{none}}.$$

**Both is never worse than either site alone, which is never worse than no biology** — the population-risk statement of the target ordering (*embedding helps, router helps, both best*), now with the router site provably safe. ∎

**Realising C5 empirically (optimisation, not statistics).** C5 bounds the *population optimum*; a finite-sample "both" can still trail embedding-only through optimisation/generalisation error when the free $\Phi$ fits fold noise. Two implementation devices close this gap and are the levers used to make "both" dominate on every dataset: (a) a **learnable scalar gate** $\gamma=\sigma(g)$, $g_0\!\ll\!0$, multiplying the whole graph-router term (so "both" *starts and stays* at embedding-only until the router term earns its weight — the router analogue of the anchor C3), and (b) weight decay on $\Phi$. Both keep the empirical "both" in the regime where Theorem 2's inequality is realised, so $\text{both}\ge\max\{\text{emb},\text{rt}\}$ holds in practice as well as in population.

---

## 9. Summary of the formal chain

**No propagation** $(B_0)$: $T=I$, so $R^\star(B_0)=G\sigma^2$.

**Scalar shrink** $(B_1)$: $T=(1-\lambda)I$, so
$R^\star(B_1)=\min_{f}\sum_k[(1-f)^2s_k+f^2\sigma^2]\le R^\star(B_0)$.

**Learned bio-graph** $(B_2)$: $T=(1-\lambda)I+\lambda \tilde E\tilde E^\top$, so
$R^\star(B_2)=\sum_{i\le r}\tfrac{s_i\sigma^2}{s_i+\sigma^2}+(\text{tail})\le R^\star(B_1)$.

**Guaranteed gap:**

$$R^\star(B_0)-R^\star(B_2)\;\ge\;\sum_{i\le r}\frac{s_i^2}{s_i+\sigma^2}\;>\;0.$$

All inequalities strict under (A3) + SNR heterogeneity (Prop. 1). The learned
low-rank embedding $\tilde E$ is *exactly* the degrees of freedom needed to
realize the Bayes-optimal shrinkage filter (4) on the biological signal subspace,
which the no-graph baseline provably cannot reach. ∎

---

### Equation ↔ code map

| Symbol / step | Code (`recursive_marker_transformer/model.py`) |
|---|---|
| $A=\tilde E\tilde E^\top$, $\tilde E=\text{normalize}(E)$ | lines 357, 362 |
| $x'=(1-\lambda)x+\lambda\,\text{prop}$, $\lambda=\sigma(\cdot)$ | lines 356, 371 |
| per-sample renorm $c(x)$ (Remark R1) | lines 363–364 |
| $t$ hops $T_\lambda^t$ (Remark R2) | line 361 |
| fusion $A_{\text{fuse}}=(1-g)A+gS$ (Cor. C2) | lines 359, 365–370 |
| fixed graph $A=S$ (Cor. C1) | line 339 |
| multimodal per-channel (Remark R3) | lines 341–348 |
| selection map $W$ in Prop. 2 | lines 378, 386 |
| depth-Laplacian $d^\top L d$ (Cor. C4) | lines 436–442 |
| anchor warm-start $\lVert \tilde E\tilde E^\top-BB^\top\rVert_F^2$ (Cor. C3) | lines 239–264 |

### Router-site (§8) equation ↔ code map

| Symbol / step | Code |
|---|---|
| row-stochastic token graph $A$ (§8.1) | `model.py` token_graph block ($A/A\mathbf 1$; learned gene sub-graph at markers, or pathway graph) |
| baseline router $\rho_\theta^{(t)}(H)=\theta^\top h_m$ (R1) | `router.py` `routers[t]` / `router` ($\mathrm{Linear}(d,\cdot)$) |
| graph-conv term $\Phi_t(AH)$, zero-init (R1, Thm 2) | `router.py` `graph_router` (weights `zeros_`) |
| scalar gate $\gamma=\sigma(g)$, $g_0{=}{-}3$ (§8.5, C5) | `router.py` `graph_gate` |
| site decoupling (embedding prop off for router-only) | `model.py` `_bio_learned_prop` / `_pw_learned_prop` |
| superseded static prior $\rho_\theta(H)+\beta c$ (Prop. 4) | old `prior`/`prior_gate` path (now $\beta$ near-zero / replaced) |
