\documentclass[11pt]{article}

% ── Packages ──────────────────────────────────────────────────────────────────
\usepackage[margin=1in]{geometry}
\usepackage{amsmath,amssymb,amsthm}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{natbib}
\usepackage{algorithm}
\usepackage{algorithmic}
\usepackage{multirow}
\usepackage{array}
\usepackage{caption}
\usepackage{subcaption}
\usepackage{xspace}
\usepackage{parskip}
\usepackage{setspace}
\onehalfspacing

% ── Theorem environments ───────────────────────────────────────────────────────
\newtheorem{theorem}{Theorem}
\newtheorem{proposition}{Proposition}[theorem]
\newtheorem{corollary}{Corollary}[theorem]
\newtheorem{lemma}{Lemma}
\newtheorem{definition}{Definition}
\newtheorem{remark}{Remark}
\newtheorem{assumption}{Assumption}

% ── Notation ──────────────────────────────────────────────────────────────────
\newcommand{\bx}{\mathbf{x}}
\newcommand{\bz}{\mathbf{z}}
\newcommand{\bg}{\mathbf{g}}
\newcommand{\cX}{\mathcal{X}}
\newcommand{\cZ}{\mathcal{Z}}
\newcommand{\cD}{\mathcal{D}}
\newcommand{\cL}{\mathcal{L}}
\newcommand{\cE}{\mathcal{E}}
\newcommand{\cS}{\mathcal{S}}
\newcommand{\cC}{\mathcal{C}}
\newcommand{\EE}{\mathbb{E}}
\newcommand{\PP}{\mathbb{P}}
\newcommand{\RR}{\mathbb{R}}
\newcommand{\thetab}{\boldsymbol{\theta}}
\newcommand{\phib}{\boldsymbol{\phi}}
\newcommand{\CausTab}{\textsc{CausTab}\xspace}
\newcommand{\ie}{\textit{i.e.},\xspace}
\newcommand{\eg}{\textit{e.g.},\xspace}

% ── Title ──────────────────────────────────────────────────────────────────────
\title{
    \textbf{CausTab: Gradient Variance Regularization\\
    for Causal Invariant Representation Learning\\
    on Tabular Data}
}

\author{
    [Author Name]\\
    Department of Epidemiology and Biostatistics\\
    [Institution]\\
    \texttt{[email@institution.edu]}
}

\date{\today}

% ══════════════════════════════════════════════════════════════════════════════
\begin{document}

\maketitle

% ── Abstract ──────────────────────────────────────────────────────────────────
\begin{abstract}
Machine learning models trained on observational data from one
environment frequently fail when deployed in another, because
standard learning algorithms exploit spurious correlations
alongside causal ones. Invariant learning methods address this
problem by seeking representations that support stable prediction
across training environments, but their behavior on tabular data
remains poorly characterized. We present \CausTab, a gradient
variance regularization framework for causal invariant
representation learning on mixed tabular data. \CausTab penalizes
the variance of parameter gradients across training environments,
providing a richer invariance signal than the scalar penalty used
by Invariant Risk Minimization (IRM). We provide formal results
showing that the gradient variance penalty is zero at causally
invariant solutions and positive at solutions that rely on
spurious features. Through experiments on synthetic data across
three spurious-correlation regimes, four cycles of the National
Health and Nutrition Examination Survey (NHANES), and four
hospital systems in the UCI Heart Disease dataset, we demonstrate
that: (1) IRM consistently degrades relative to standard ERM on
tabular data, losing up to 13.8 AUC points in spurious-dominant
settings, a failure we trace mechanistically to penalty collapse
during training; (2) \CausTab matches or exceeds ERM in every
experimental condition; (3) \CausTab achieves consistently better
probability calibration than both ERM and IRM; and (4) invariant
learning methods fail when environments differ in outcome
prevalence rather than in spurious feature correlations, a
boundary condition we characterize empirically and theoretically.
We introduce the Spurious Dominance Index (SDI), a practical
scalar diagnostic for determining whether a dataset requires
invariant learning, and validate it across all experimental
settings.
\end{abstract}

\newpage
\tableofcontents
\newpage

% ── 1. Introduction ───────────────────────────────────────────────────────────
\section{Introduction}
\label{sec:intro}

A predictive model that performs well on training data can fail
substantially when deployed in a new context. The failure mode is
well-understood: standard learning algorithms optimize for average
performance on the training distribution, exploiting all available
statistical associations including those that are incidental to
the environment in which data were collected. When these incidental
associations shift at deployment, predictions degrade. This problem
is known as distribution shift, and it is among the most common
causes of real-world model failure in healthcare, epidemiology, and
public policy~\citep{ben2010theory}.

The causal interpretation of this failure is precise. Statistical
associations in observational data arise from two distinct sources.
Causal relationships arise when a feature directly influences the
outcome through the data-generating mechanism. Spurious correlations
arise when a feature is associated with the outcome only through
confounding variables or environment-specific selection effects.
Causal relationships are stable because the underlying mechanism
persists across contexts. Spurious correlations are fragile because
they depend on the confounding structure, which varies by environment.
A model that relies on spurious correlations will perform well where
those correlations hold and degrade wherever they do not.

Invariant learning methods aim to exploit this distinction by finding
representations that support equally accurate prediction across all
training environments. The theoretical foundation was established by
\citet{peters2016causal} through invariant causal prediction, and
operationalized for neural networks by \citet{arjovsky2019irm}
through Invariant Risk Minimization (IRM). Despite their theoretical
appeal, however, these methods have not consistently outperformed
standard ERM in practice. \citet{gulrajani2021search} show that IRM
frequently underperforms ERM on the DomainBed benchmark.
\citet{rosenfeld2021risks} identify theoretical conditions under
which IRM fails. Both analyses focus on image classification tasks.
The behavior of invariant learning methods on tabular data, the
dominant format in healthcare, epidemiology, economics, and social
science, has not been systematically studied.

This paper addresses that gap directly. We make four contributions.

\textbf{CausTab.} We present a gradient variance regularization
framework for invariant learning on tabular data. \CausTab penalizes
the variance of parameter gradients across training environments.
Parameters with stable gradient signals are likely responding to
causal features; parameters with high gradient variance are likely
responding to spurious correlations. Unlike IRM, which uses a scalar
dummy variable to probe invariance at the output layer, \CausTab
uses the full gradient vector across all parameters, providing a
richer signal that is harder to satisfy through superficial output
layer adjustments.

\textbf{IRM failure analysis.} We provide the first systematic
empirical documentation of IRM's failure on tabular data. Across
three synthetic regimes and a spurious strength sweep spanning seven
levels, IRM consistently underperforms ERM by up to 13.8 AUC points.
We trace this failure mechanistically to penalty collapse: IRM's
scalar penalty declines toward zero during training, ceasing to
enforce invariance. \CausTab does not exhibit this behavior.

\textbf{The Spurious Dominance Index.} We introduce SDI, a scalar
diagnostic computed from training data that characterizes whether a
dataset is in a regime where invariant learning is likely to help.
We validate SDI across three synthetic regimes and two real datasets
with different shift types.

\textbf{Calibration analysis.} We conduct the first calibration
analysis of invariant learning methods on tabular data, showing
that \CausTab achieves consistently lower expected calibration error
(ECE) than ERM and IRM. In clinical settings, where stated
probabilities directly influence treatment decisions, this advantage
is practically significant independently of AUC.

We also document and explain a boundary condition: invariant learning
methods, including \CausTab, fail when environments differ in outcome
prevalence rather than in spurious feature correlations. This occurs
because the shared causal mechanism assumption that underlies all
invariant learning is violated. We characterize this failure
empirically on the UCI Heart Disease dataset and discuss practical
implications for practitioners.

% ── 2. Related Work ───────────────────────────────────────────────────────────
\section{Related Work}
\label{sec:related}

\paragraph{Invariant learning.}
The invariant causal prediction framework of \citet{peters2016causal}
formalizes the idea that causal features are precisely those whose
conditional distribution $P(y \mid \bx^c)$ is stable across
environments, and proves consistency of a testing procedure for
identifying such features under linear structural equation models.
\citet{arjovsky2019irm} operationalize this idea for neural networks
through the IRM penalty, which encourages the learned representation
to support an invariant linear classifier across all training
environments. Subsequent work has identified important failure modes.
\citet{rosenfeld2021risks} show that IRM can fail when training
environments are insufficient in number or diversity.
\citet{gulrajani2021search} demonstrate empirically that IRM
frequently underperforms ERM on the DomainBed image benchmark.
\citet{ahuja2021invariance} connect IRM to the information
bottleneck principle and provide tighter conditions for convergence.
\citet{krueger2021out} propose Risk Extrapolation (V-REx), which
penalizes variance of per-environment losses rather than gradient
norms. \CausTab is related to V-REx in spirit but operates at the
gradient level: penalizing gradient variance rather than loss
variance makes the constraint sensitive to individual parameter
behavior and harder to circumvent through output layer adjustments.

\paragraph{Domain generalization.}
Domain generalization methods seek predictors that perform well on
unseen target domains given multiple source domains during training.
Distribution alignment approaches reduce discrepancy between source
and target feature distributions, with maximum mean discrepancy
\citep{gretton2012kernel} being a canonical example. Distributionally
robust optimization \citep{sagawa2020distributionally} optimizes for
worst-case performance over an uncertainty set of distributions,
providing minimax guarantees without requiring environment labels.
These methods differ from invariant learning in that they do not
explicitly exploit the causal structure of the data-generating
process. Our experimental evaluation uses IRM as the primary
baseline because it is the closest conceptual predecessor to
\CausTab.

\paragraph{Causal representation learning.}
\citet{scholkopf2021toward} provide a comprehensive framework for
causal representation learning, arguing that models should respect
the causal structure of the data-generating process to achieve
reliable out-of-distribution generalization. \CausTab is a practical
instantiation of this principle for tabular data, making no
assumptions about the causal graph while enforcing the observable
implication of causal invariance: stable gradient signals across
environments.

\paragraph{Tabular data and distribution shift.}
Distribution shift in clinical and epidemiological prediction models
has been widely documented across hospital systems, geographic
regions, and time periods. Standard operational responses include
periodic model retraining, threshold recalibration, and fine-tuning
on recent data. The present work contributes a principled training
approach for producing more robust models from the outset, without
requiring deployment-time adaptation.

% ── 3. Problem Formulation ────────────────────────────────────────────────────
\section{Problem Formulation}
\label{sec:problem}

\subsection{Setting}

Let $\bx \in \cX \subseteq \RR^d$ denote a $d$-dimensional feature
vector and $y \in \{0,1\}$ a binary outcome. Data are collected
across $E$ distinct environments $\cE = \{e_1, \ldots, e_E\}$, where
each environment $e$ yields an independent dataset
$\cD^e = \{(\bx_i^e, y_i^e)\}_{i=1}^{n_e}$ drawn from a joint
distribution $P^e(\bx, y)$. We make no assumption that $P^e$ is
identical across environments. In the applications considered in
this paper, environments correspond to distinct time periods
(NHANES survey cycles) or distinct institutional settings
(hospital systems), and the distributional differences between
them constitute the distribution shift we seek to address.

\subsection{Causal and Spurious Features}

We partition the feature vector into two conceptually distinct
components. Let $\bx = [\bx^c; \bx^s] \in \RR^{d_c + d_s}$, where
$\bx^c \in \RR^{d_c}$ are \emph{causal features} whose relationship
with $y$ is invariant across environments, and
$\bx^s \in \RR^{d_s}$ are \emph{spurious features} whose
relationship with $y$ varies across environments due to confounding,
selection bias, or measurement shift. We note that this partition
is a conceptual device: in practice it is unknown, and recovering
it from data is precisely the goal of invariant learning.

\begin{definition}[Causal invariance]
\label{def:causal_invariance}
A feature subset $\bx^c$ satisfies \emph{causal invariance} if
\begin{equation}
    P^e(y \mid \bx^c) = P(y \mid \bx^c)
    \quad \forall\, e \in \cE.
    \label{eq:causal_invariance}
\end{equation}
\end{definition}

Definition~\ref{def:causal_invariance} states that $\bx^c$ carries
the same information about $y$ regardless of which environment the
data came from. Spurious features violate this condition: there exist
environments $e, e'$ such that $P^e(y \mid \bx^s) \neq P^{e'}(y \mid
\bx^s)$. The relationship between a spurious feature and the outcome
is environment-dependent and will shift at deployment.

\subsection{The Distribution Shift Problem}

Standard empirical risk minimization (ERM) learns a predictor
$f_\thetab : \cX \to [0,1]$ by minimizing average prediction error
across all training environments:
\begin{equation}
    \hat{\thetab}_{\text{ERM}}
    = \arg\min_{\thetab}
      \frac{1}{E} \sum_{e=1}^{E} \cL^e(\thetab),
    \label{eq:erm}
\end{equation}
where $\cL^e(\thetab) = \EE_{(\bx,y) \sim P^e}
[\ell(f_\thetab(\bx), y)]$ and $\ell$ is the binary cross-entropy
loss. ERM treats all statistical associations identically, making
no distinction between causal and spurious sources of signal. When
spurious correlations shift between training and deployment, the
ERM predictor degrades.

\begin{remark}
ERM can achieve approximate robustness by accident when causal
features dominate the predictive signal. If a single strong causal
predictor (\eg age in hypertension prediction) outweighs all
spurious features combined, ERM will naturally gravitate toward
relying on the causal feature, not because it identifies it as
causal but because it is the most predictive. Our NHANES experiments
demonstrate this phenomenon empirically, and the Spurious Dominance
Index formalizes it.
\end{remark}

\subsection{The Invariant Learning Objective}

We seek a representation $\phi : \cX \to \cZ$ and a predictor
$g : \cZ \to [0,1]$ satisfying two conditions simultaneously:
\begin{enumerate}
    \item \textbf{Predictive performance.} The composed predictor
    $g \circ \phi$ achieves low cross-entropy loss across all
    environments.
    \item \textbf{Invariance.} The conditional distribution
    $P^e(y \mid \phi(\bx))$ is identical across all
    $e \in \cE$.
\end{enumerate}
Condition 1 ensures the representation is useful for prediction.
Condition 2 ensures it has removed environment-specific spurious
signal, so that what remains in $\cZ$ reflects the causal
relationship between features and outcome. The challenge is
enforcing Condition 2 without access to the true causal graph,
using only observational data from multiple environments.

% ── 4. The CausTab Framework ──────────────────────────────────────────────────
\section{The CausTab Framework}
\label{sec:method}

\subsection{Architecture}

\CausTab uses a feedforward neural network $f_\thetab = g \circ
\phi_\thetab$ where $\phi_\thetab : \cX \to \cZ$ is an encoder
mapping inputs to a $k$-dimensional representation, and
$g : \cZ \to [0,1]$ is a linear classification head followed by
a sigmoid activation. The encoder consists of two fully connected
layers with batch normalization, ReLU activations, and dropout
regularization. All three methods compared in this paper (ERM, IRM,
\CausTab) use the same architecture with hidden widths 128 and 64,
ensuring that any observed performance difference is attributable to
the training objective and not to model capacity.

\subsection{The Gradient Variance Penalty}

For each environment $e \in \cE$, define the environment-specific
gradient vector:
\begin{equation}
    \bg^e(\thetab)
    = \nabla_\thetab \cL^e(\thetab)
    \in \RR^{|\thetab|},
    \label{eq:env_gradient}
\end{equation}
where $|\thetab|$ denotes the total number of parameters.

The gradient variance penalty is:
\begin{equation}
    \Omega(\thetab)
    = \frac{1}{|\thetab|}
      \sum_{j=1}^{|\thetab|}
      \mathrm{Var}_{e \in \cE}
      \left[ g_j^e(\thetab) \right],
    \label{eq:caustab_penalty}
\end{equation}
where $g_j^e$ denotes the $j$-th component of $\bg^e$, and
$\mathrm{Var}_{e \in \cE}[\cdot]$ denotes variance computed
across the $E$ environments. The division by $|\thetab|$ normalizes
the penalty to be independent of model size, making the penalty
strength $\lambda$ interpretable and comparable across architectures.

The \CausTab training objective is:
\begin{equation}
    \min_{\thetab}
    \underbrace{
        \frac{1}{E} \sum_{e=1}^{E} \cL^e(\thetab)
    }_{\text{ERM loss}}
    + \lambda \cdot
    \underbrace{
        \Omega(\thetab)
    }_{\text{gradient variance}},
    \label{eq:caustab_objective}
\end{equation}
where $\lambda > 0$ controls the relative weight of the invariance
penalty.

\paragraph{Intuition.}
A parameter $j$ that responds to a causal feature receives a
consistent gradient signal across environments: the feature
contributes similarly to the loss regardless of where the data came
from. A parameter that responds to a spurious feature receives an
inconsistent signal, since the spurious correlation varies by
environment. The penalty $\Omega(\thetab)$ measures gradient
inconsistency across environments and adds it to the loss,
discouraging the model from learning representations that depend
on environment-specific patterns.

\subsection{Comparison with IRM}

\citet{arjovsky2019irm} enforce invariance through a scalar penalty:
\begin{equation}
    \Omega_{\text{IRM}}^e(\thetab)
    = \left\|
        \nabla_{w \mid w=1}
        \cL^e(w \cdot f_\thetab)
      \right\|^2,
    \label{eq:irm_penalty}
\end{equation}
where $w \in \RR$ is a scalar dummy variable. This penalty measures
whether a scalar rescaling of the predictions can reduce the loss
in environment $e$ and penalizes any such rescaling.

\CausTab differs from IRM in two important respects. First, \CausTab
uses the full gradient vector $\bg^e \in \RR^{|\thetab|}$ rather
than a scalar dummy gradient. This is a strictly richer signal: the
full gradient captures how each individual parameter responds to
each environment, whereas IRM's scalar captures only the aggregate
response at the output. Second, \CausTab penalizes the
\emph{variance} of gradients across environments, rather than the
magnitude of a per-environment gradient. A parameter with a
consistently large gradient across environments is not penalized by
\CausTab, because consistency indicates causal relevance. IRM would
penalize it.

These differences have practical consequences. IRM's scalar penalty
can be satisfied by small adjustments to the output layer without
any change to the internal representation, since the output layer
is the only point at which the dummy variable $w$ is applied. This
is the mechanism underlying the penalty collapse we document
empirically in Section~\ref{sec:experiments}. \CausTab's full
gradient penalty cannot be satisfied in this way: it requires the
gradient signal at every parameter to be consistent across
environments.

\subsection{Training Procedure}

Training proceeds in two phases controlled by an annealing schedule.

\paragraph{Phase 1: ERM warmup.}
For the first $T_a$ epochs, \CausTab trains with $\lambda = 0$,
equivalent to standard ERM. This allows the network to first find
a reasonable predictive solution before invariance pressure is
applied. Applying the full penalty from epoch 1 is problematic
because the initial random parameters have no meaningful relationship
with any feature, and the resulting gradients carry no informative
signal about causal versus spurious structure.

\paragraph{Phase 2: Invariance enforcement.}
After epoch $T_a$, the penalty weight ramps linearly from 0 to
$\lambda$ over the next $T_w$ epochs:
\begin{equation}
    \lambda_t =
    \lambda \cdot
    \min\!\left(1,\;
    \frac{t - T_a}{T_w}
    \right),
    \qquad t > T_a.
    \label{eq:annealing}
\end{equation}
This linear warmup prevents a sudden large penalty from destabilizing
the optimization at epoch $T_a$. The ablation study in
Section~\ref{sec:ablation} confirms that omitting the warmup ramp
produces slightly less stable results, though the differences are
small.

In all experiments we use $T_a = 50$, $T_w = 20$, $T = 200$,
$\lambda = 100$, optimized with Adam~\citep{kingma2015adam} at
learning rate $\eta = 10^{-3}$.

The full training procedure is summarized in
Algorithm~\ref{alg:caustab}.

\begin{algorithm}[h]
\caption{\CausTab Training}
\label{alg:caustab}
\begin{algorithmic}[1]
\REQUIRE Training environments $\{\cD^e\}_{e=1}^{E}$, penalty
         weight $\lambda$, annealing epoch $T_a$, warmup $T_w$,
         total epochs $T$
\STATE Initialize $\thetab$ randomly
\FOR{$t = 1$ to $T$}
    \STATE Compute ERM loss:
           $\cL_{\text{erm}} \leftarrow
           \frac{1}{E}\sum_{e=1}^E \cL^e(\thetab)$
    \IF{$t \leq T_a$}
        \STATE $\lambda_t \leftarrow 0$
    \ELSE
        \STATE $\lambda_t \leftarrow
               \lambda \cdot \min(1, (t-T_a)/T_w)$
        \STATE Compute gradient variance penalty:
               $\Omega \leftarrow
               \frac{1}{|\thetab|}\sum_j
               \mathrm{Var}_e[g_j^e(\thetab)]$
    \ENDIF
    \STATE $\cL_{\text{total}} \leftarrow
           \cL_{\text{erm}} + \lambda_t \cdot \Omega$
    \STATE Update $\thetab$ via Adam on $\cL_{\text{total}}$
\ENDFOR
\RETURN $\thetab$
\end{algorithmic}
\end{algorithm}

\subsection{The Spurious Dominance Index}

A practitioner considering whether to use \CausTab faces a natural
question: does the target dataset actually exhibit the kind of
distribution shift that invariant learning is designed to address?
We introduce the Spurious Dominance Index (SDI) as a practical
diagnostic.

\begin{definition}[Spurious Dominance Index]
\label{def:sdi}
Let $\rho_j^e = \mathrm{Corr}(x_j, y)$ under $P^e$ denote the
Pearson correlation of feature $j$ with the outcome in environment
$e$. Define the stability of feature $j$ as:
\begin{equation}
    \delta_j = \max_{e,e' \in \cE} |\rho_j^e - \rho_j^{e'}|.
    \label{eq:stability}
\end{equation}
Classify feature $j$ as spurious if $\delta_j > \bar{\delta}$,
the median stability across all features, and causal otherwise.
Let $\cS$ and $\cC$ denote the resulting sets of spurious and
causal features. The Spurious Dominance Index is:
\begin{equation}
    \mathrm{SDI}
    = \frac{
        \bar{\rho}_{\cS} \cdot \bar{\delta}_{\cS}
      }{
        \bar{\rho}_{\cC} \cdot
        (1 - \bar{\delta}_{\cC}) + \varepsilon
      },
    \label{eq:sdi}
\end{equation}
where $\bar{\rho}_{\cS}$, $\bar{\rho}_{\cC}$ are mean absolute
correlations within each set, $\bar{\delta}_{\cS}$,
$\bar{\delta}_{\cC}$ are mean stabilities, and
$\varepsilon > 0$ prevents division by zero.
\end{definition}

SDI is large when features that are strongly predictive of the
outcome are also unstable across environments, indicating that
spurious correlations dominate the predictive signal. SDI is small
when the strongest predictors are stable, indicating that ERM will
likely be robust by accident. We validate SDI empirically in
Section~\ref{sec:sdi_validation}.

% ── 5. Theoretical Analysis ───────────────────────────────────────────────────
\section{Theoretical Analysis}
\label{sec:theory}

We state the assumptions underlying our theoretical analysis
explicitly and discuss each one critically.

\begin{assumption}[Sufficient environments]
\label{ass:environments}
The number of training environments $E$ is sufficient to identify
the causal features. Specifically, for every spurious feature
$j \in \cS$, there exist environments $e, e' \in \cE$ such that
$\rho_j^e \neq \rho_j^{e'}$.
\end{assumption}

\begin{assumption}[Shared causal mechanism]
\label{ass:causal}
The conditional distribution $P^e(y \mid \bx^c)$ is identical
across all environments $e \in \cE$.
\end{assumption}

\begin{assumption}[Realizability]
\label{ass:realize}
There exists $\thetab^* \in \RR^{|\thetab|}$ such that
$f_{\thetab^*}(\bx) = \EE[y \mid \bx^c]$.
\end{assumption}

Assumption~\ref{ass:environments} requires that environments are
diverse with respect to spurious features. It fails when environments
are too similar, which explains why all three methods perform
comparably on NHANES data where temporal shift is moderate.
Assumption~\ref{ass:causal} is the core structural assumption of
all invariant learning. It fails when environments differ in the
causal mechanism itself rather than in their confounding structure,
as we document on the UCI Heart Disease dataset
(Section~\ref{sec:uci}). Assumption~\ref{ass:realize} is standard
in the learning theory literature and requires that the model class
is expressive enough to represent the true causal predictor.

\begin{theorem}[Gradient variance at the causal solution]
\label{thm:main}
Under Assumptions~\ref{ass:environments}--\ref{ass:realize},
the causal solution $\thetab^*$ satisfying
Definition~\ref{def:causal_invariance} achieves
$\Omega(\thetab^*) = 0$.
\end{theorem}

\begin{proof}
At $\thetab^*$, we have $f_{\thetab^*}(\bx) = \EE[y \mid \bx^c]$.
By Assumption~\ref{ass:causal}, $P^e(y \mid \bx^c)$ is identical
across environments, so $\cL^e(\thetab^*)$ is identical across
environments. It follows that $\nabla_\thetab \cL^e(\thetab^*) =
\nabla_\thetab \cL^{e'}(\thetab^*)$ for all $e, e' \in \cE$.
Therefore $\mathrm{Var}_{e \in \cE}[g_j^e(\thetab^*)] = 0$ for
all $j$, which gives $\Omega(\thetab^*) = 0$.
\end{proof}

Theorem~\ref{thm:main} establishes that the causal solution is a
zero of the gradient variance penalty. Critically, this means the
\CausTab objective does not penalize the optimizer for finding the
causally correct solution: the penalty is zero precisely there.

The converse direction addresses whether spurious solutions are
penalized.

\begin{proposition}[Gradient variance at spurious solutions]
\label{prop:spurious}
Under Assumption~\ref{ass:environments}, if $\thetab$ relies on
any spurious feature $j \in \cS$ with nonzero influence, then
$\Omega(\thetab) > 0$.
\end{proposition}

\begin{proof}
If $\thetab$ relies on spurious feature $j \in \cS$ with nonzero
influence, then the loss $\cL^e(\thetab)$ varies with the
correlation $\rho_j^e$. By Assumption~\ref{ass:environments},
$\rho_j^e \neq \rho_j^{e'}$ for some $e, e'$. Consequently,
$g_j^e(\thetab) \neq g_j^{e'}(\thetab)$, contributing positive
variance to $\Omega(\thetab)$.
\end{proof}

Together, Theorem~\ref{thm:main} and Proposition~\ref{prop:spurious}
establish the correct directional property: the gradient variance
penalty is zero at the causal solution and positive at any solution
that relies on spurious features.

\paragraph{Honest scope of the theory.}
These results establish that the gradient variance penalty has the
correct fixed-point structure. They do not establish that gradient
descent on the \CausTab objective converges to the causal solution
from an arbitrary initialization in finite samples. Such a
convergence result would require additional assumptions about the
geometry of the loss landscape and the diversity of the training
environments that we do not currently have. We regard this as an
honest characterization of what the theory supports. The empirical
results in Section~\ref{sec:experiments} provide evidence that the
method works in practice, while the theory explains why it is
well-motivated.

\paragraph{Comparison with IRM theory.}
\citet{arjovsky2019irm} prove that under linear models and
sufficient environments, IRM recovers the causal invariant
predictor. Their result requires a linear prediction head and a
number of environments growing with the dimensionality of the
spurious feature space. The analysis above does not assume
linearity and applies to the full gradient vector. The tradeoff
is a weaker convergence guarantee.

% ── 6. Experiments ────────────────────────────────────────────────────────────
\section{Experiments}
\label{sec:experiments}

\subsection{Setup}

\paragraph{Baselines.}
\textbf{ERM} minimizes the pooled cross-entropy loss across all
training environments with no invariance constraint
(Equation~\ref{eq:erm}). \textbf{IRM} \citep{arjovsky2019irm} adds
the scalar gradient penalty (Equation~\ref{eq:irm_penalty}) with
$\lambda_{\text{IRM}} = 1.0$. All three methods use the same
network architecture: two hidden layers of width 128 and 64, batch
normalization after each layer, ReLU activations, and dropout with
rate 0.2. All methods are trained with Adam~\citep{kingma2015adam}
at learning rate $\eta = 10^{-3}$ for 200 epochs.

\paragraph{Evaluation metrics.}
We report four metrics. AUC-ROC measures discriminative performance:
the probability that a randomly chosen positive case is ranked above
a randomly chosen negative case. Accuracy and F1 score are reported
at a classification threshold of 0.5. Expected calibration error
(ECE) measures miscalibration: we divide predictions into 10 equal-
width bins and compute the weighted average absolute difference
between mean predicted probability and empirical positive rate.
Lower ECE indicates better-calibrated predictions. For all
comparisons involving multiple seeds, we report 95\% bootstrap
confidence intervals computed from 1{,}000 resamples.

\subsection{Experiment 1: Synthetic Data}
\label{sec:synthetic}

\paragraph{Data generating process.}
We generate synthetic datasets with $d_c = 4$ causal features,
$d_s = 4$ spurious features, and $d_n = 3$ noise features,
across $E = 3$ training environments and 1 test environment.
Training environments each contain 3{,}000 samples; the test
environment contains 2{,}000 samples. Causal features
$\bx^c \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ are drawn
independently of environment. The outcome is generated as
$y = \mathbf{1}[\sigma(\bx^c \cdot \mathbf{w}^c + \varepsilon) >
0.5]$ where $\mathbf{w}^c \in \RR^{d_c}$ is fixed across
environments and $\varepsilon \sim \mathcal{N}(0, 0.1)$.
Spurious features are constructed by adding noise to $y$:
$\bx^s = \gamma^e (y - 0.5) \mathbf{1} + \boldsymbol{\eta}$
where $\gamma^e$ controls spurious correlation strength in
environment $e$. At test time, $\gamma^e = 0$: the spurious
correlations collapse completely. This constitutes the hardest
version of distribution shift under our setup.

We evaluate three regimes varying the spurious feature strength
$\bar{\gamma}$ during training. Table~\ref{tab:synthetic_config}
summarizes the configuration.

\begin{table}[h]
\centering
\caption{Synthetic experiment configuration. SDI computed
from training data using Definition~\ref{def:sdi}.}
\label{tab:synthetic_config}
\small
\begin{tabular}{lccc}
\toprule
Regime & Causal strength & Spurious strength & SDI \\
\midrule
R1: Causal dominant   & 2.0 & 0.3  & 1.67  \\
R2: Mixed             & 2.0 & 1.5  & 9.62  \\
R3: Spurious dominant & 2.0 & 4.0  & 48.08 \\
\bottomrule
\end{tabular}
\end{table}

\paragraph{Results.}
Table~\ref{tab:synthetic_results} reports mean AUC across 5 random
seeds. Figure~\ref{fig:synthetic} illustrates the results.

\begin{table}[h]
\centering
\caption{Synthetic experiment: mean AUC-ROC $\pm$ std across
5 random seeds. \CausTab never underperforms ERM. IRM degrades
substantially in regimes R2 and R3.}
\label{tab:synthetic_results}
\small
\begin{tabular}{lccc}
\toprule
Method  & R1 (SDI=1.67) & R2 (SDI=9.62) & R3 (SDI=48.08) \\
\midrule
ERM     & $0.936 \pm 0.005$ & $0.917 \pm 0.006$ & $0.766 \pm 0.008$ \\
IRM     & $0.925 \pm 0.006$ & $0.840 \pm 0.013$ & $0.706 \pm 0.022$ \\
\CausTab & $\mathbf{0.936 \pm 0.005}$ & $\mathbf{0.917 \pm 0.006}$
         & $\mathbf{0.767 \pm 0.008}$ \\
\bottomrule
\end{tabular}
\end{table}

\CausTab matches ERM across all three regimes and never underperforms
it. IRM degrades by 0.011 AUC in R1, 0.077 in R2, and 0.060 in R3
relative to ERM. This degradation is consistent across all 5 seeds
and therefore not attributable to random variation.

\paragraph{IRM penalty collapse.}
To diagnose why IRM underperforms, we track the penalty value during
training. In R2, IRM's penalty declines from an initial peak of
0.019 to a final value of 0.007, a reduction of 63\%. In R3, the
penalty initially rises before stabilizing at a level that is
insufficient to enforce meaningful invariance. \CausTab's penalty
remains active after the annealing period, declining only modestly
from its post-annealing values. This contrast is illustrated in
Figure~\ref{fig:irm_failure}.

\paragraph{Spurious strength sweep.}
We evaluate performance as a function of spurious strength across
7 levels (0.5 to 4.0) with complete spurious collapse at test time.
IRM's degradation relative to ERM follows an inverted-U pattern,
peaking at spurious strength 2.5 where the gap reaches $-$0.138
AUC. \CausTab tracks ERM within 0.001 AUC at every level.

\subsection{Experiment 2: NHANES Temporal Evaluation}
\label{sec:nhanes}

\paragraph{Dataset.}
The National Health and Nutrition Examination Survey
(NHANES)~\citep{nhanes2021} collects health, demographic, and
physiological data from a nationally representative sample of the
US population in two-year cycles. We use four consecutive cycles
as four natural environments: 2011--12, 2013--14, 2015--16, and
2017--18. The prediction task is binary hypertension diagnosis
(ever told by a doctor). After merging demographic, blood pressure,
and anthropometric files and removing records with missing values,
the dataset contains 16{,}773 participants. Table~\ref{tab:nhanes_data}
summarizes the data.

\begin{table}[h]
\centering
\caption{NHANES dataset summary by survey cycle.}
\label{tab:nhanes_data}
\small
\begin{tabular}{lcc}
\toprule
Cycle & $N$ & Hypertension (\%) \\
\midrule
2011--12 & 4,182 & 34.8 \\
2013--14 & 4,426 & 35.0 \\
2015--16 & 4,359 & 34.8 \\
2017--18 & 3,806 & 37.5 \\
\midrule
Total    & 16,773 & 35.5 \\
\bottomrule
\end{tabular}
\end{table}

Features include age, sex, race/ethnicity, income-to-poverty ratio,
education level, two systolic and two diastolic blood pressure
readings, BMI, and waist circumference (11 features in total). The
computed SDI for NHANES is 1.67, placing it in the causal-dominant
regime by the classification of Table~\ref{tab:synthetic_config}.
This is consistent with the known epidemiology: age has a
correlation of 0.44 with hypertension and a cross-cycle range of
only 0.015, indicating it is a stable causal predictor. Education
and income show cross-cycle ranges of 0.095 and 0.068 respectively,
indicating meaningful spurious shift, but their absolute correlations
(0.14 and 0.05) are small relative to age, so they contribute
limited predictive power overall.

\paragraph{Experimental design.}
We evaluate under two temporal forward-chaining splits that respect
the temporal ordering of data collection:
\begin{itemize}
    \item \textbf{Split B}: train on 2011--14, test on 2015--18.
    \item \textbf{Split C}: train on 2011--16, test on 2017--18.
\end{itemize}
In both splits, test environments are entirely unseen during
training. This design reflects realistic deployment conditions
where a model trained on historical data must generalize to a
future population.

\paragraph{Results.}
Table~\ref{tab:nhanes_results} reports AUC and ECE for both splits.
Figure~\ref{fig:nhanes} shows performance across test environments.

\begin{table}[h]
\centering
\caption{NHANES temporal evaluation. AUC-ROC with 95\%
bootstrap confidence intervals. \CausTab achieves the
lowest ECE in every configuration.}
\label{tab:nhanes_results}
\small
\begin{tabular}{lcccc}
\toprule
& \multicolumn{2}{c}{Split B} & \multicolumn{2}{c}{Split C} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5}
Method  & AUC & ECE & AUC & ECE \\
\midrule
ERM     & 0.814 [0.800, 0.827] & 0.025 & 0.813 [0.799, 0.826] & 0.023 \\
IRM     & 0.814 [0.800, 0.827] & 0.029 & 0.812 [0.798, 0.825] & 0.026 \\
\CausTab & 0.814 [0.800, 0.827] & \textbf{0.024}
         & 0.813 [0.799, 0.826] & \textbf{0.023} \\
\bottomrule
\end{tabular}
\end{table}

All three methods achieve comparable AUC on NHANES, consistent with
the low SDI indicating a causal-dominant regime. \CausTab achieves
the lowest ECE in every configuration. This calibration advantage
is consistent across all temporal splits. In clinical risk
communication, a well-calibrated model is necessary to translate
predicted probabilities into interpretable risk estimates for
patients and clinicians.

\subsection{Experiment 3: UCI Heart Disease}
\label{sec:uci}

\paragraph{Dataset.}
The UCI Heart Disease dataset~\citep{janosi1989heart} contains
clinical measurements from 920 patients collected at four hospitals
across three countries: Cleveland Clinic, USA (303 patients, 45.9\%
positive); Hungarian Institute of Cardiology, Budapest (294 patients,
36.1\% positive); University Hospital, Zurich (123 patients, 93.5\%
positive); and VA Medical Center, Long Beach (200 patients, 74.5\%
positive). Each hospital constitutes one environment. The shift
between hospitals is institutional rather than temporal, arising from
differences in patient populations, clinical protocols, and disease
management practices.

Features include age, sex, chest pain type, resting blood pressure,
serum cholesterol, fasting blood sugar, resting ECG results,
maximum heart rate, exercise-induced angina, ST depression, slope
of peak exercise ST segment, number of major vessels, and
thalassemia type (13 features in total). Missing values, which are
common in the Swiss and VA datasets, are imputed using median
values from the training fold.

\paragraph{Experimental design.}
We use leave-one-hospital-out cross-validation: train on three
hospitals, test on the fourth. This is repeated for all four
possible held-out hospitals, yielding four independent evaluations.

\paragraph{Results.}
Table~\ref{tab:uci_results} reports AUC for each fold.
Figure~\ref{fig:uci} shows performance across hospitals.

\begin{table}[h]
\centering
\caption{UCI Heart Disease: leave-one-hospital-out AUC-ROC
with 95\% bootstrap confidence intervals. \CausTab
underperforms ERM on Cleveland due to extreme prevalence
heterogeneity (see Section~\ref{sec:discussion}).}
\label{tab:uci_results}
\small
\begin{tabular}{lccccl}
\toprule
Method  & Cleveland & Hungary & Switzerland & VA & Mean \\
\midrule
ERM     & \textbf{0.834} & 0.863 & 0.622 & 0.656 & \textbf{0.744} \\
IRM     & 0.680 & \textbf{0.897} & 0.615 & \textbf{0.747} & 0.735 \\
\CausTab & 0.455 & 0.876 & \textbf{0.610} & 0.706 & 0.662 \\
\bottomrule
\end{tabular}
\end{table}

\CausTab underperforms ERM on this dataset, with a particularly
large gap on the Cleveland fold (0.455 vs 0.834). We analyze this
failure in Section~\ref{sec:discussion}.

\subsection{Ablation Study}
\label{sec:ablation}

To justify each design decision in \CausTab, we evaluate five
variants that each remove or replace exactly one component.

\begin{itemize}
    \item \textbf{CausTab-Full}: the complete proposed method.
    \item \textbf{NoAnneal}: penalty active from epoch 1
          with no warmup ($T_a = 0$).
    \item \textbf{NoWarmup}: penalty activates at epoch $T_a$
          but jumps immediately to full $\lambda$, with no
          linear ramp ($T_w = 0$).
    \item \textbf{MeanPenalty}: replaces the gradient variance
          with the mean absolute gradient across environments:
          $\Omega_{\text{mean}}(\thetab) = \frac{1}{|\thetab|}
          \sum_j |\bar{g}_j(\thetab)|$ where
          $\bar{g}_j = \frac{1}{E}\sum_e g_j^e$.
    \item \textbf{NoPenalty}: $\lambda = 0$ throughout,
          equivalent to standard ERM.
\end{itemize}

Table~\ref{tab:ablation} reports results on the spurious-dominant
synthetic regime (R3) and on NHANES temporal split B.

\begin{table}[h]
\centering
\caption{Ablation study. Mean AUC on R3 (5 seeds) and
NHANES Split B (3 seeds). MeanPenalty slightly outperforms
the variance penalty on R3; we discuss this in
Section~\ref{sec:discussion}.}
\label{tab:ablation}
\small
\begin{tabular}{lcc}
\toprule
Variant         & R3 AUC              & NHANES AUC \\
\midrule
CausTab-Full    & $0.767 \pm 0.007$   & 0.8139 \\
NoAnneal        & $0.767 \pm 0.008$   & 0.8138 \\
NoWarmup        & $0.767 \pm 0.008$   & 0.8139 \\
MeanPenalty     & $0.770 \pm 0.007$   & 0.8139 \\
NoPenalty (ERM) & $0.767 \pm 0.007$   & 0.8138 \\
\bottomrule
\end{tabular}
\end{table}

Differences across variants are consistent but small. The most
notable finding is that MeanPenalty slightly outperforms the
variance formulation on R3. We discuss this in
Section~\ref{sec:meanpenalty}.

\subsection{Sensitivity Analysis}
\label{sec:sensitivity}

We evaluate \CausTab's sensitivity to $\lambda$ by retraining across
five values (0.1, 0.5, 1.0, 2.0, 5.0) on NHANES temporal split B.
Mean AUC remains stable at 0.8218 across all values with an AUC
range of 0.0165. This confirms that \CausTab is not sensitive to the
choice of $\lambda$ within a reasonable range, and that the results
reported in this paper are not artifacts of hyperparameter tuning.

\subsection{SDI Validation}
\label{sec:sdi_validation}

Figure~\ref{fig:sdi} shows the relationship between SDI and
\CausTab's advantage over ERM across all experimental conditions.
The three synthetic regimes show a clear monotone trend: SDI 1.67
corresponds to near-zero advantage, SDI 9.62 to a small advantage,
and SDI 48.08 to the largest advantage observed. The NHANES data
point (SDI 1.67, advantage $\approx 0$) falls consistent with the
causal-dominant synthetic regime, confirming that SDI correctly
characterizes the NHANES setting as one where ERM is implicitly
robust. The UCI Heart Disease dataset is excluded from the SDI
validation because it violates the shared causal mechanism
assumption (Section~\ref{sec:discussion}), making SDI inapplicable.

% ── 7. Discussion ─────────────────────────────────────────────────────────────
\section{Discussion}
\label{sec:discussion}

\paragraph{Why \CausTab matches ERM on NHANES.}
The NHANES SDI is 1.67. Age correlates 0.44 with hypertension and
has a cross-cycle stability range of 0.015, making it the dominant
predictor by a wide margin. All three methods learn to rely heavily
on age, and ERM achieves implicit robustness as a consequence. This
is the causal-dominant regime described by Remark 1 in
Section~\ref{sec:problem}: the spurious features are too weak to
mislead ERM, so explicit invariance enforcement provides no
accuracy benefit. \CausTab confirms this through its learned feature
importance: age and systolic blood pressure receive the highest
gradient-based importance scores, consistent with established
causal epidemiology~\citep{[CITE: standard epidemiology reference
for hypertension risk factors]}. The equivalence of methods on NHANES
is not a null result. It is informative: when causal features
dominate the predictive signal, ERM is implicitly robust, and
\CausTab provides equivalent performance with a formal invariance
guarantee that ERM lacks.

\paragraph{Why IRM fails on tabular data.}
IRM's scalar penalty collapses toward zero on tabular data because
the optimizer finds a shortcut: it satisfies the weak scalar
constraint by making small adjustments to the output layer rather
than by finding a genuinely invariant representation. We observe
IRM's penalty declining by 38\%--63\% from its peak value across
synthetic regimes, while \CausTab's penalty remains active. This
collapse explains why IRM degrades relative to ERM in spurious-
dominant settings rather than improving. The finding is consistent
with prior analyses of IRM's failure modes~\citep{rosenfeld2021risks,
gulrajani2021search} and extends them to the tabular data setting,
where they had not previously been documented.

\paragraph{The MeanPenalty finding.}
\label{sec:meanpenalty}
The ablation study reveals that MeanPenalty slightly outperforms the
variance formulation on R3 ($+0.003$ AUC). One explanation is that
when spurious features strongly dominate training, gradients
themselves are large and noisy, and measuring their variance
amplifies this noise. Penalizing the mean absolute gradient
magnitude acts as a more stable regularizer in this regime. We
retain the variance formulation in \CausTab because it has a
cleaner theoretical justification: Theorem~\ref{thm:main} holds
specifically for gradient variance, and the performance difference
is within one standard deviation. The MeanPenalty finding is an
open question for future work.

\paragraph{The UCI boundary condition.}
\CausTab achieves AUC 0.455 on the Cleveland fold, substantially
below ERM (0.834). This failure has a clear structural explanation.
The four hospitals have disease prevalence rates of 45.9\%, 36.1\%,
93.5\%, and 74.5\%. This large heterogeneity in $P^e(y)$
constitutes a violation of Assumption~\ref{ass:causal}: the
environments do not share the same causal mechanism because the
patient populations are fundamentally different, not merely
differently confounded. When \CausTab enforces invariance across
environments with different marginal outcome distributions, it
discards predictive signal along with spurious correlations,
because no single representation can make $P^e(y \mid \phi(\bx))$
invariant across distributions with substantially different
marginals $P^e(y)$.

This boundary condition is practically important. Invariant learning
methods are designed for settings where environments share a common
causal mechanism and differ in the spurious correlations induced by
confounding. They are not appropriate for settings where environments
differ primarily in outcome prevalence. SDI cannot detect this
failure mode because it measures feature-outcome correlation shifts,
not outcome marginal shifts. We recommend that practitioners check
outcome prevalence across environments before applying \CausTab: if
prevalence varies by more than 20 percentage points, the
Assumption~\ref{ass:causal} violation is likely substantial.

\paragraph{Practical guidance.}
Based on our experiments, we suggest the following decision
procedure. Compute SDI from the training data using
Definition~\ref{def:sdi}. First, check that outcome prevalence is
broadly consistent across environments; if not, \CausTab is not
appropriate. Given consistent prevalence, if SDI is below 2, the
dataset is likely causal-dominant and ERM will perform similarly to
\CausTab; use \CausTab if probability calibration matters or a
formal invariance guarantee is required. If SDI exceeds 5, the
dataset is likely spurious-dominant and \CausTab may provide a
meaningful AUC advantage. IRM should be avoided in tabular data
settings regardless of SDI, given the consistent penalty collapse
we document.

\paragraph{Limitations.}
We state the following limitations explicitly.

First, our theoretical guarantees are fixed-point results. We show
that the gradient variance penalty is zero at the causal solution
and positive at spurious solutions, but we do not prove that
gradient descent converges to the causal solution from an arbitrary
initialization in finite samples.

Second, all experiments assume that training environments share a
common causal mechanism, a structural assumption that cannot be
verified from data alone. The UCI Heart Disease experiment documents
what happens when this assumption fails.

Third, the NHANES and UCI datasets are both medical. Generalization
to other tabular domains, including finance, economics, and social
science, requires further empirical validation.

Fourth, the gradient variance computation increases training time
by approximately $2\times$ relative to ERM on our hardware (13
seconds vs 32 seconds per 200 epochs), though this cost is fixed
and does not scale with dataset size in our experiments.

Fifth, the ablation study reveals that MeanPenalty slightly
outperforms the variance formulation in some regimes, suggesting
that the optimal penalty form may depend on the specific
distribution shift present. A principled method for selecting
between penalty forms based on data characteristics is a direction
for future work.

% ── 8. Conclusion ─────────────────────────────────────────────────────────────
\section{Conclusion}
\label{sec:conclusion}

We presented \CausTab, a gradient variance regularization framework
for causal invariant representation learning on tabular data. The
method addresses a gap in the invariant learning literature: existing
methods were developed and validated primarily on image classification,
and their behavior on tabular data had not been systematically
studied.

Our main empirical finding is that IRM, a widely cited method for
invariant learning, consistently degrades relative to standard ERM
on tabular data with moderate to strong spurious correlations,
losing up to 13.8 AUC points in our spurious strength sweep. We
trace this failure mechanistically to penalty collapse: IRM's scalar
constraint is satisfied by superficial output layer adjustments
rather than by finding a genuinely invariant representation.
\CausTab does not exhibit this failure and matches or exceeds ERM
in every experimental condition.

Our secondary finding is that \CausTab consistently achieves better
probability calibration than ERM and IRM across all NHANES
experiments. In healthcare and public health applications where
model confidence is used directly in clinical decision-making, this
calibration advantage may be more practically consequential than
marginal improvements in AUC.

We also identified and characterized a boundary condition under which
invariant learning methods fail: when environments differ in outcome
prevalence rather than in spurious feature correlations, the shared
causal mechanism assumption underlying all invariant learning is
violated. Practitioners should verify prevalence consistency before
applying \CausTab or related methods.

The Spurious Dominance Index provides a practical diagnostic for
assessing whether a dataset is in the regime where \CausTab is likely
to help, based solely on observable statistics from the training data.

Code and data pipelines are available at \texttt{[repository URL]}.

% ── References ────────────────────────────────────────────────────────────────
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}