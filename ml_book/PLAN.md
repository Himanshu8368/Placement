# ML Book — Plan (v4)

Status: **approved structure, being written. This is a comprehensive guide, not a short textbook.** v3 (2026-10-25) added the missing topics, made ML systems a reading map into CS249r, and sorted the resources you collected. v4 (2026-10-25) groups the parts into **5 volumes**, reorders them by dependency (RL before LLMs, theory after the classical models), and adds a written **Agentic Systems** part.

---

## 1. Why the book was rebuilt (v1 → v2)

The old 18-volume book (`archive/v1/`) had these problems:

- Each major topic got a single page, so it was too shallow.
- Every chapter used the same 12-field template, full of meta-advice about lab notes.
- None of the code was ever run.
- The content lived in escaped Python strings, which made it hard to edit.
- There were 292 hard-coded "Volume NN" cross-references, and several pointed to the wrong place.
- Some text was garbled.
- There were no worked examples, exercises, or interview questions.
- Modern topics such as LoRA, DPO, RAG, and the KV cache were missing.

v2 is written by hand as Quarto chapters with executable code. The chapter format is in `STYLE.md`.

---

## 2. Scope decisions

| Decision | Why |
|---|---|
| **ML systems is not written. CS249r is used as is** (Part XIII is a reading map). | CS249r already covers this ground in depth: Vol 1 (single machine), Vol 2 (scale, distributed, inference, privacy, robustness, responsible AI, sustainability), Vol 3 (agentic and LLM systems), and Vol 4 (physical AI). Writing our own would duplicate it at lower quality. |
| Model compression, MLOps, serving, hardware, distributed training, privacy, security, and responsible AI → CS249r | These are systems topics. The book keeps only the algorithmic core that interviews ask about: LLM quantization math, the KV cache, speculative decoding (Part XII), and distillation (Part VII). |
| **Agentic systems are written** as Part XIII (12 chapters), with CS249r Vol 3 as the main text | Agents are now central to ML engineering and interviews, so they get a full part. CS249r Vol 3 is the deep reference. The part adds runnable agent loops, worked examples, exercises, and interview Q&A, which CS249r does not have. |
| A Tools part is added | Every later chapter assumes NumPy, pandas, and PyTorch fluency. |
| A learning-theory part is added | It was missing (GT CS7545, Caltech LFD, UML). |
| Probabilistic modelling is expanded | HMMs, Kalman filters, graphical models, MCMC, and VI were missing or only one line. |
| Classical CV, classical NLP, speech/audio, 3D vision, SSMs, MoE, reasoning models, multimodal LLMs, flow matching, model-based RL, and imitation learning are added | These were missing from v2's 76-chapter outline. |

---

## 3. Outline: 5 volumes, 16 parts, 155 chapters (plus CS249r Vol 1–2 bundled; Vol 3–4 to plug in later)

The book is one Quarto project. Volumes group the parts in the table of contents, and each volume can also be rendered as its own PDF. Parts are ordered so that every chapter only depends on earlier ones:

```
Vol 1 Foundations ──► Vol 2 Classical ML ──► Vol 3 Deep Learning ──► Vol 4 RL · LLMs · Agents ──► Vol 5 Applications & Career
 Tools, Math           Supervised, Practice,   DL, Vision, NLP,         RL → LLMs → Agentic         Specialized, Systems map,
                       Probabilistic, Theory   Generative               systems                     Interview
```

Changes from v3:

1. **RL now comes before LLMs**, because RLHF, PPO, GRPO and RLVR build on policy gradients.
2. **"The learning problem" (ERM, bias–variance) opens supervised learning.** The formal theory (PAC, VC, Rademacher, online learning) moves after the classical models, so it has concrete examples to work with.
3. **There is a new written Part XIII, Agentic Systems** (12 chapters). Agents move out of the LLM part.
4. **Folders are named by topic** (`math/`, `supervised/`, …) rather than by part number, so reordering never forces a rename.
5. **v4.1 (comprehensive-guide pass).** Chapters that held two big topics are split: linear algebra, probability, optimization, boosting, GPs, metrics, detection, diffusion, LLM data/scaling, RLHF/DPO, inference, embeddings/RAG. Missing topics are added: Bayesian nonparametrics, PAC-Bayes, NLP tasks, multi-agent RL, distribution shift, continual learning, meta-learning, multi-task learning, and production debugging scenarios. Every chapter must meet the depth standard in `STYLE.md` (25–60 pages, ≥15 exercises, ≥12 interview questions).

★ marks a full from-scratch implementation, and ✅ marks a chapter that is already written.

---

### VOLUME 1: FOUNDATIONS

#### Part I: Tools (3) · `tools/`
1. Python and NumPy for ML: vectorisation, broadcasting, views and copies, numerical stability ★
2. Data wrangling and exploratory analysis: pandas, plotting, distributions, correlations
3. PyTorch fundamentals: tensors, autograd, `nn.Module`, `Dataset`/`DataLoader`, devices, a first training loop ★

#### Part II: Mathematical foundations (13) · `math/`
1. ✅ Linear algebra I: vectors, norms, matrices as maps, subspaces, rank, projections, least squares, QR, conditioning ★ (written. Its eigen/SVD sections move to II.2 and are expanded there)
2. Linear algebra II, matrix decompositions and identities: eigendecomposition, SVD, Cholesky, LU, the Woodbury identity, Schur complement, block matrices, Kronecker products, trace and log-det tricks ★
3. Matrix calculus and automatic differentiation: gradients, Jacobians, Hessians, identities, forward and reverse mode ★
4. Probability foundations: sample spaces, random variables, conditioning and Bayes' rule, expectation, variance and covariance, change of variables, LLN, CLT
5. Probability distributions reference: discrete and continuous families, moments, how they relate, the exponential family, a conjugacy table ★
6. Multivariate distributions and the Gaussian: covariance, marginals and conditionals, Gaussian identities, Mahalanobis distance ★
7. Statistical inference: estimators, bias–variance of estimators, MLE, MAP, confidence intervals, bootstrap, hypothesis tests ★
8. Bayesian inference: conjugate priors, posterior predictive, model evidence ★
9. Convex analysis and duality: convex sets and functions, Lagrangian, KKT, strong duality
10. First- and second-order optimization: GD convergence (smooth, strongly convex), line search, Newton, BFGS/L-BFGS, conjugate gradient, projected, proximal and coordinate methods ★
11. Stochastic optimization: SGD theory, minibatching, variance reduction (SVRG), momentum and Nesterov, AdaGrad/RMSProp/Adam, the non-convex landscape ★
12. Information theory: entropy, cross-entropy, KL divergence, mutual information, MLE as KL minimisation
13. Numerical computation and Monte Carlo: floating point, log-sum-exp, sampling, importance sampling ★

### VOLUME 2: CLASSICAL MACHINE LEARNING

#### Part III: Supervised learning (15) · `supervised/`
1. The learning problem: ERM, train and test, bias–variance decomposition, approximation vs estimation error, no free lunch
2. Linear regression: three derivations, the probabilistic view, inference on coefficients, basis features ★
3. Regularization and feature selection: ridge, lasso (coordinate descent), elastic net, subset selection ★
4. Logistic regression and GLMs: softmax, Newton/IRLS, the exponential family ★
5. Perceptron and linear classifiers: mistake bound, hinge loss, multiclass strategies ★
6. Generative classifiers: Naive Bayes, LDA, QDA, generative vs discriminative ★
7. Nearest neighbours and nonparametric methods: k-NN, kernel regression, the curse of dimensionality ★
8. Kernel methods: the kernel trick, RKHS, the representer theorem, kernel ridge regression ★
9. Support vector machines: hard and soft margin, the dual, SMO, SVR ★
10. Decision trees: CART, impurity measures, pruning, categorical and missing values ★
11. Bagging and random forests: variance reduction, OOB error, feature importance ★
12. Boosting I: AdaBoost, forward stagewise additive modelling, deriving gradient boosting ★
13. Boosting II, XGBoost, LightGBM, and CatBoost: second-order splits, histograms, leaf-wise growth, ordered boosting, tuning ★
14. Bayesian linear regression and the evidence approximation ★
15. Gaussian processes: kernels, GP regression, hyperparameter learning, GP classification, scalable GPs ★

#### Part IV: ML in practice (10) · `practice/`
1. An end-to-end ML project: framing, baselines, pipelines, iteration (Hands-On ML, AAAMLP, FSDL lectures 1 and 8)
2. Feature engineering and data cleaning: encoding, scaling, missing data, outliers, text and date features, leakage
3. Classification metrics: confusion matrix, precision/recall/F-score, ROC and PR curves, AUC, multiclass averaging, cost-sensitive thresholds ★
4. Regression, ranking, and probabilistic metrics: MSE/MAE/MAPE, quantile loss, NDCG/MAP/MRR, log loss, Brier score, proper scoring rules ★
5. Validation and hyperparameter tuning: k-fold, grouped and time-series CV, nested CV, random, Bayesian and Hyperband search
6. Imbalanced and noisy data: resampling, class weights, threshold tuning, label noise
7. Calibration, uncertainty, and conformal prediction ★
8. Comparing models and A/B testing: paired tests, McNemar, power, multiple testing, sequential tests
9. Interpretability: permutation importance, PDP/ICE, LIME, SHAP (including TreeSHAP), counterfactuals
10. Learning with few labels: semi-supervised learning, active learning, weak supervision

#### Part V: Unsupervised learning and probabilistic models (10) · `probabilistic/`
1. Clustering: k-means and k-means++, hierarchical, DBSCAN, spectral, cluster evaluation ★
2. Gaussian mixtures and the EM algorithm, with the general EM derivation ★
3. Linear latent-variable models: PCA, probabilistic PCA, factor analysis, ICA, NMF ★
4. Manifold learning and visualisation: kernel PCA, MDS, Isomap, t-SNE, UMAP
5. Density estimation and anomaly detection: KDE, isolation forest, one-class SVM ★
6. Probabilistic graphical models: Bayesian networks, d-separation, MRFs, factor graphs, belief propagation
7. Sequential latent-variable models: HMMs (forward–backward, Viterbi, Baum–Welch), the Kalman filter ★
8. Markov chain Monte Carlo: Metropolis–Hastings, Gibbs sampling, HMC, convergence diagnostics ★
9. Variational inference: the ELBO, mean field, CAVI, stochastic VI, the reparameterisation trick ★
10. Bayesian model selection and nonparametrics: evidence, BIC/AIC, the Dirichlet process, the Chinese restaurant process, DP mixtures

#### Part VI: Learning theory (4) · `theory/`
1. PAC learning and VC dimension: finite hypothesis classes, growth function, Sauer's lemma, sample complexity
2. Rademacher complexity, margin bounds, and algorithmic stability
3. Online learning: regret, the perceptron and halving algorithms, multiplicative weights, online-to-batch conversion
4. PAC-Bayes and generalization of modern models: PAC-Bayes bounds, compression bounds, why uniform bounds fail for deep networks

### VOLUME 3: DEEP LEARNING

#### Part VII: Deep learning foundations (12) · `deep/`
1. Neural networks: perceptron to MLP, activations, universal approximation (*By hand*)
2. Backpropagation and autograd, building a micrograd-style engine ★
3. Loss functions and output layers
4. Initialization and signal propagation: Xavier and He, vanishing and exploding gradients ★
5. Optimizers for deep learning: SGD with momentum, Adam/AdamW, LR schedules, warm-up, clipping ★
6. Normalization: BatchNorm, LayerNorm, GroupNorm, RMSNorm ★
7. Regularization: weight decay, dropout, data augmentation, early stopping, label smoothing, mixup
8. Residual connections and architecture design: depth, skip connections, gating (*By hand*: ResNet block)
9. The training recipe: troubleshooting and testing (FSDL lecture 3), overfitting one batch, tuning, mixed precision
10. Generalization in deep learning: double descent, implicit bias, sharpness, an intuition for the NTK
11. Transfer learning, fine-tuning, and knowledge distillation ★
12. Representation learning: self-supervised and contrastive learning (InfoNCE), metric learning, autoencoders (*By hand*) ★

#### Part VIII: Computer vision (10) · `vision/`
1. Images and classical vision: image formation, filtering, the Fourier view, edges, HOG and SIFT ★
2. Convolutional layers: arithmetic, padding, stride, dilation, pooling, receptive fields (*By hand*) ★
3. CNN architectures: LeNet to ResNet, EfficientNet, ConvNeXt ★
4. Object detection I: IoU, NMS, mAP, R-CNN → Faster R-CNN, FPN, SSD, YOLO, RetinaNet (focal loss) ★
5. Object detection II: anchor-free detectors (FCOS, CenterNet), DETR and its successors, open-vocabulary detection (OWL-ViT, Grounding DINO)
6. Segmentation: FCN, U-Net (*By hand*), Mask R-CNN, SAM ★
7. Vision transformers: ViT (*By hand*), Swin, DeiT ★
8. Self-supervised vision and vision–language: SimCLR, MAE, DINO, CLIP
9. Video understanding: 3D convolutions, two-stream networks, video transformers
10. 3D vision: camera geometry, stereo and epipolar geometry, NeRF, Gaussian splatting

#### Part IX: NLP and sequence models (10) · `nlp/`
1. Classical NLP: preprocessing, bag-of-words and TF-IDF, n-gram language models and smoothing ★
2. Word embeddings: word2vec (skip-gram with negative sampling), GloVe, fastText ★
3. Sequence labelling: POS tagging and NER, HMM taggers, CRFs, BiLSTM-CRF ★
4. Recurrent networks: RNN (*By hand*), backpropagation through time, LSTM, GRU ★
5. Seq2seq, attention, and machine translation: Bahdanau attention, beam search, BLEU ★
6. The Transformer: scaled dot-product and multi-head attention, positional encoding, the full block (*By hand*) ★
7. Pretrained language models: BERT, GPT, T5, masked vs causal vs span objectives
8. NLP tasks: question answering, summarization, information extraction, entity linking
9. Build a GPT from scratch (nanoGPT-style on tiny Shakespeare) ★
10. Decoding and text-generation evaluation: greedy, beam, top-k/top-p, temperature, perplexity, ROUGE, BERTScore

#### Part X: Generative models (10) · `generative/`
1. Overview of generative modelling and autoregressive models: PixelCNN, WaveNet ★
2. Variational autoencoders: ELBO, reparameterisation, β-VAE, VQ-VAE ★
3. GANs: minimax objective, DCGAN, WGAN, mode collapse (*By hand*) ★
4. Normalizing flows: change of variables, RealNVP, Glow ★
5. Energy-based models and score matching: denoising score matching, Langevin dynamics ★
6. Diffusion models I: DDPM forward and reverse processes, the ELBO derivation, noise prediction, DDIM ★
7. Diffusion models II: the score-SDE view, the probability-flow ODE, fast samplers (DPM-Solver), consistency models and distillation ★
8. Flow matching and rectified flow ★
9. Guidance, latent diffusion, and text-to-image/video: classifier-free guidance, Stable Diffusion, DiT
10. Evaluating generative models: likelihood, FID, IS, precision/recall, CLIP score

### VOLUME 4: REINFORCEMENT LEARNING, LLMs, AND AGENTS

#### Part XI: Reinforcement learning (10) · `rl/`
1. Multi-armed bandits: ε-greedy, UCB, Thompson sampling, contextual bandits ★
2. MDPs and dynamic programming: Bellman equations, value and policy iteration ★
3. Monte Carlo and temporal-difference learning: SARSA, Q-learning, TD(λ) ★
4. Function approximation and deep Q-learning: DQN, Double and Dueling DQN, experience replay ★
5. Policy gradients and actor-critic: REINFORCE, baselines, A2C, GAE ★
6. Advanced policy optimization: TRPO, PPO, DDPG, TD3, SAC ★
7. Model-based RL and planning: MCTS, AlphaZero, MuZero, world models
8. Imitation learning and inverse RL: behaviour cloning, DAgger, GAIL
9. Multi-agent RL and game theory: Nash equilibria, self-play, independent learners, centralised training with decentralised execution
10. Offline RL and the bridge to LLMs: CQL, IQL, a preview of RLHF and RLVR

#### Part XII: Large language models (18) · `llm/`
1. Tokenization: BPE (from scratch), WordPiece, Unigram, byte-level ★
2. Pretraining data: sources, quality filtering, deduplication, mixtures, synthetic data, contamination ★
3. Scaling laws and compute: FLOP and memory arithmetic, Kaplan and Chinchilla, emergent abilities, training stability ★
4. Modern LLM architecture: pre-norm, RMSNorm, SwiGLU, RoPE, GQA/MQA ★
5. Efficient attention and long context: FlashAttention tiling, sparse and linear attention, context extension ★
6. Mixture of experts: routing, load balancing, expert parallelism ★
7. State-space models and alternatives: S4, Mamba, RWKV, hybrids
8. Supervised fine-tuning and PEFT: instruction tuning, LoRA, QLoRA, adapters ★
9. RLHF: reward modelling (Bradley–Terry), PPO for language models, the KL penalty, reward hacking ★
10. The DPO family: DPO derivation, IPO, KTO, ORPO, SimPO, online and iterative DPO ★
11. Reasoning and test-time compute: chain-of-thought, self-consistency, process vs outcome rewards, RLVR, GRPO ★
12. LLM inference I: KV cache, continuous batching, paged attention, speculative decoding ★
13. LLM inference II: quantization (INT8/INT4, GPTQ, AWQ, SmoothQuant), pruning and distillation of LLMs ★
14. Text embeddings and vector search: contrastive embedding training, HNSW, IVF-PQ, ANN trade-offs ★
15. Retrieval-augmented generation: chunking, hybrid retrieval, reranking, query rewriting, GraphRAG, RAG evaluation ★
16. Multimodal LLMs: vision encoders, LLaVA-style connectors, audio
17. Evaluating LLMs: benchmarks, contamination, LLM-as-judge, human evaluation
18. Interpretability and safety: probing, mechanistic interpretability, sparse autoencoders, jailbreaks, red-teaming

#### Part XIII: Agentic systems (12) · `agents/`
The main text is **CS249r Vol 3**. Supporting sources are listed in §4. Each chapter uses a small, **deterministic agent loop** written in code, driven by a scripted or mock policy so it runs offline, plus optional cells that call a real API.

1. What is an agent: the LLM-in-a-loop model, workflows vs agents (prompt chaining, routing, parallelisation, orchestrator–workers, evaluator–optimizer), levels of autonomy
2. Prompting and context engineering: prompt structure, few-shot, structured output, context budgets, compaction and summarisation, prompt caching (CS249r Vol 3 ch. 4)
3. Tool use and function calling: JSON schemas, the tool-call loop, error handling, code execution, the Model Context Protocol (MCP) ★ (CS249r Vol 3 ch. 7)
4. Reasoning and planning patterns: ReAct, plan-and-execute, reflection/Reflexion, tree search over thoughts, using test-time compute ★ (CS249r Vol 3 ch. 3)
5. Memory: short-term scratchpads, long-term and episodic memory, memory retrieval, agentic RAG ★ (CS249r Vol 3 ch. 6)
6. The agent harness: state machines and graphs, durable execution, checkpointing, retries, failure recovery ★ (CS249r Vol 3 ch. 10–12)
7. Multi-agent systems: orchestrator–worker, handoffs, debate, role specialisation, coordination costs (CS249r Vol 3 ch. 18)
8. Coding, browser, and computer-use agents: SWE-agent-style agent–computer interfaces, sandboxes, GUI agents (CS249r Vol 3 ch. 8)
9. Evaluating agents: SWE-bench, WebArena, τ-bench, GAIA, trajectory-level metrics, pass^k, cost and latency (CS249r Vol 3 ch. 13)
10. Training agents: trajectory curation, SFT on traces, multi-turn RL and RLVR for agents (CS249r Vol 3 ch. 15–17)
11. Agent security and safety: prompt injection, tool permissions, sandboxing, human-in-the-loop, guardrails (CS249r Vol 3 ch. 9)
12. Agent operations and economics: tracing and observability, cost models, caching, routing between models (CS249r Vol 3 ch. 14, 19)

### VOLUME 5: APPLICATIONS AND CAREER

#### Part XIV: Specialized topics (10) · `specialized/`
1. Recommender systems: collaborative filtering, matrix factorization (ALS), two-tower models, sequential recommenders ★
2. Learning to rank and search: pointwise, pairwise and listwise losses, LambdaMART, BM25 vs dense retrieval ★
3. Time series and forecasting: ARIMA, exponential smoothing, state-space models, deep forecasting, backtesting ★
4. Graph neural networks: spectral basics, GCN, GraphSAGE, GAT, message passing ★
5. Causal inference: potential outcomes, DAGs, confounding, propensity scores, uplift modelling ★
6. Speech and audio: signals, the Fourier transform, spectrograms and MFCCs, CTC, Whisper ★
7. Distribution shift and domain adaptation: covariate and label shift, importance weighting, DANN, test-time adaptation ★
8. Continual learning: catastrophic forgetting, EWC, replay, parameter isolation
9. Meta-learning and few-shot learning: MAML, prototypical networks, in-context learning as meta-learning ★
10. Multi-task and multi-label learning: hard and soft sharing, loss balancing, label dependencies

#### Part XV: ML systems, CS249r plugged in (3 written guide chapters + CS249r volumes) · `systems/`
The official CS249r releases are **bundled** in `books/cs249r/` (fetched by `scripts/fetch_cs249r.sh`) and copied into the built site. Every chapter guide deep-links to PDF pages (`#page=N`), to the online version, and to the EPUB.

1. ✅ ML Systems with CS249r: how CS249r is organised, the iron law and the Fleet Law (with runnable estimates), reading paths, and a map from this book's chapters to CS249r chapters
2. ✅ CS249r Vol 1 guide (16 chapters, release v0.8.0): page links, what to master, the matching chapters here, and interview self-checks for each chapter
3. ✅ CS249r Vol 2 guide (17 chapters, preview v0.3.0): the same format
4. *(later)* CS249r Vol 3 guide (Agentic ML Systems): plugged in when the authors release it. Until then, Part XIII covers agents
5. *(later)* CS249r Vol 4 guide (Physical AI Systems): plugged in when the authors release it (optional)

| Topic | CS249r |
|---|---|
| ML lifecycle, data engineering, MLOps | Vol 1 ch. 3, 4, 14 |
| How NNs compute, frameworks, training systems | Vol 1 ch. 5, 7, 8 |
| Data selection, compression (pruning, quantization, distillation) | Vol 1 ch. 9, 10 |
| Hardware acceleration, benchmarking, serving | Vol 1 ch. 11, 12, 13 |
| Compute infrastructure, networks, storage | Vol 2 ch. 2, 3, 4 |
| Collectives, distributed training, fault tolerance, orchestration | Vol 2 ch. 5, 6, 7, 8 |
| Fleet efficiency, serving at scale, edge and federated learning, ops at scale | Vol 2 ch. 9, 10, 11, 12 |
| Security and privacy, robustness, sustainability, responsible AI | Vol 2 ch. 13–16 (+ Vol 1 ch. 15) |
| Agentic systems | **Part XIII of this book** now. CS249r Vol 3 later |
| Robotics and physical AI | CS249r Vol 4 later (optional) |

#### Part XVI: Interview preparation (5) · `interview/`
1. ML system design: the framework and 9 case studies (feed ranking, search, ads CTR, recommendations, fraud, moderation, RAG assistant, **customer-support agent**, ETA)
2. ML coding drills: about 40 "implement X from scratch" problems with tests (k-means, logistic regression, attention, BPE, an agent loop, …) ★
3. Question bank: about 330 fundamentals questions with answers, grouped by part
4. ML debugging and production scenarios: "accuracy dropped after deploy", "offline gain, online loss", leakage hunts, latency regressions, worked as case studies
5. Probability, statistics, and ML-math puzzles

**Appendices** (`appendix/`): A. Notation · B. NumPy/PyTorch cheat sheet · C. Table of distributions · D. Glossary · E. Reading map (which free-book chapter goes with which book chapter)

---

## 4. Resources — what we use and where

Local copies are in `sources/` (see `sources/README.md`). Books and courses are used for derivations, figures, and **exercises/homework**. Lecture videos are listed only as optional companions.

### Books (primary)
| Book | Used in |
|---|---|
| MML (Deisenroth et al.) | Part I |
| Murphy — PML1, PML2; MLAPP (2012, "book 0", your own copy) | I, III, V, X, everywhere probabilistic |
| ISLP, ESL | III, IV |
| UDL (Prince) | VII, VIII, X, XII |
| D2L | I, VII–IX (code) |
| Bishop PRML / *Deep Learning: Foundations and Concepts* | III, V, X |
| Shalev-Shwartz & Ben-David (UML), Boyd & Vandenberghe, MacKay | VI, II.7–II.9 |
| *Machine Learning Refined* (Watt et al.) | II.8, III (clear optimization-first explanations) |
| Szeliski, *Computer Vision: Algorithms and Applications* 2e | VIII.1, VIII.9 |
| SLP3 (Jurafsky & Martin) | IX, XIV.6 |
| Sutton & Barto | XI |
| *Hands-On ML* 3e notebooks (Géron), *Approaching (Almost) Any ML Problem* (Thakur), *Python Data Science Handbook* (VanderPlas) | I, IV |
| CS249r | XV (used directly), XIII (Vol 3 = main text for agents) |
| GenAI Handbook (genai-handbook.github.io) | XII roadmap and reading lists |
| Chip Huyen, *ML Interviews* | XVI + interview Q&A in every chapter |

### Courses (notes, slides, homework → exercises)
| Area | Courses kept |
|---|---|
| Math for ML | Tübingen *Mathematics for ML* (notes + solutions), CMU 10-601 math homework, GT CS4641 (Gopalan) |
| Classical ML | CMU 10-601, Berkeley CS189 (Shewchuk notes), Stanford CS229, UW CSE446, UBC CPSC 340/540 (Schmidt), Cornell CS5785 Applied ML (Kuleshov notebooks), Sarath Chandar ML notes |
| Learning theory | GT CS7545, Caltech *Learning from Data* |
| Deep learning | CMU 11-785 (F25), UMich EECS 498/598 (Johnson), GT CS7643, UIUC CS444, Karpathy *Zero to Hero*, Stanford CS25 |
| Vision | Stanford CS231n (notes + assignments), CS231a (3D), UMich EECS 442, MIT 6.801 (classical) |
| NLP | Lena Voita's NLP course, CMU ANLP (Neubig), Stanford CS224n, UW CSE447/517, IITD COL772 |
| LLMs | Stanford CS336 (LM from scratch), Raschka *LLMs from Scratch*, Karpathy LLM101n, Waterloo CS886, GT CS8803-LLM, IITD ELL881, Stanford CS229S |
| Generative | CMU 10-799 *Diffusion & Flow Matching* |
| Agentic systems | **CS249r Vol 3** (main text); Hugging Face *Agents Course* (`sources/agents-course`); Microsoft *AI Agents for Beginners* (`sources/ai-agents-for-beginners`); MCP spec (`sources/modelcontextprotocol`); Anthropic engineering posts (*Building Effective Agents*, context engineering, multi-agent research system); Lilian Weng, *LLM Powered Autonomous Agents*; Berkeley *LLM Agents* MOOC (CS294/194-196); OpenAI *A Practical Guide to Building Agents*; papers: ReAct, Toolformer, Reflexion, SWE-agent, Voyager, τ-bench |
| Efficient ML | MIT 6.5940 (Han Lab), used only for XII.12–13, everything else → CS249r |
| Visual explainers | colah's blog, Jay Alammar's *Illustrated Transformer*, Transformer Explainer, *Illustrated ML* |
| "By hand" worked examples | Tom Yeh, *AI by Hand* / *Deep Learning by Hand 2026* (byhand.ai): linear algebra, NN, MLP, SVM, autoencoder, and soon GAN, ResNet, RNN, Transformer, CNN, U-Net, ViT. Its method is the model for the **By hand** example in every architecture chapter (see `STYLE.md`). Videos and Excel workbooks only, so it is not stored locally. |
| ML product and engineering practice | Full Stack Deep Learning (`sources/the-full-stack-website`): 2022 course lectures 1, 3, 4, 6, 8 → IV.1 (end-to-end project), VII.9 (troubleshooting and testing), XVI.1 (system design). LLM Bootcamp 2023 (prompt engineering, augmented LMs, LLMOps, UX) → XII.15 (RAG), XIII.2–3, and the XVI.1 RAG case study. FSDL's deployment, monitoring, and infra lectures date from 2022, so CS249r (2026) is used for those instead. |

### Dropped (and why)
| Resource | Reason |
|---|---|
| Data Engineering Zoomcamp, UW CSE412/442/512, GT CSE6242, GT 8803 (DB) | Databases, data viz, and data engineering are out of scope (or covered by CS249r Vol 1 ch. 4) |
| UW CSE160/163/180, Illinois Discovery, CodeSignal, datasciencecourse.net, bcfoltz Statistics 101 | Intro programming and intro stats, already covered by Part 0 + PDSH + Part I |
| Awesome lists (awesome-ml, TopDeepLearning, awesome-cv, speech list, 365-days CV), deep-learning-drizzle, Robbie Allen 2018 list, donnemartin and yogeshhk notebooks | Link directories with no content of their own, or outdated |
| MIT 6.S191 (2020), VT ECE5984 (2015), Princeton COS597G (2022) | Superseded by newer courses in the same area |
| Signal processing courses (Hacettepe BBM413/CMP717, DSP Fundamentals) | Only the Fourier basics are needed, covered in VIII.1 and XIV.6 |
| UIUC CV (Forsyth), Cornell CS6670, UW CSE455/576/P576, Brown CSCI1430/2951-T, Dellaert 4476, *Advances in Vision*, dl4cv-nptel, jalalirs CV | Overlap with CS231n + Szeliski + EECS 442. Keeping fewer, better sources |
| GT DL for robotics, Stanford CS348k | Robotics and visual-computing systems → CS249r Vol 4 if ever needed |
| UW CSE543, Bo Dai's courses, lowyx GT notes, Brown DL, Raissi Applied DL | Overlap with kept DL courses (CSE543 can be an optional extra for VII.10) |
| Applied Language Technology MOOC, Kochmar NLP, GT CS7650, IITB CS626, CS447 (UIUC), CSCI1460 | Overlap with Voita + ANLP + CS224n |
| Individual YouTube videos and playlists | Not checked. Used as optional viewing only, never as a source |

**Your own PDFs** (MLAPP, *ml_exercise_solution.pdf*, Andrew Ng DL notes, AAAMLP, cheat sheets, *vorlesung_main.pdf*): put them in `sources/pdf/personal/` and they will be used for exercises and cross-checks.

---

## 5. Chapter format

See `STYLE.md`. In short: Sources box → why it matters → derivations → from-scratch code with `assert` checks against libraries → ≥2 worked examples per idea → ≥8 exercises [E/M/H] with collapsible solutions → ≥8 interview Q&A → key-equations summary. Code runs during the render (`freeze: auto`). Cross-references use Quarto labels.

## 6. Execution

1. Volume 1: Part II (Math, 1 of 13 done; Chapter 1 to be split and expanded to the depth standard), then Part I (Tools).
2. Then Volumes 2 → 5 in order. Each part is written by at most 2–3 parallel writers, then verified with `scripts/check_chapter.py`, a full render, and a check of the math and cross-references.
3. Commit after each part. Each volume gets its own PDF when it is complete.
