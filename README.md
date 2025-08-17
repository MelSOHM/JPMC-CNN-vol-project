# JPMC-CNN-vol-project

Four references : 

>[1] Jiang, J., Kelly, B. and Xiu, D. (2023) (Re-)Imag(in)ing Price Trends. J Finance, 78: 3193-3249. \
>[2] He, K., Zhang, X., Ren, S., and Sun, J. (2016) Deep Residual Learning for Image Recognition. IEEE Conference on Computer Vision and Pattern Recognition (CVPR). \
>[3] Szegedy, C., Ioffe, S., Vanhoucke, V., and Alemi, A. (2017) Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning. AAAI.\
>[4] Bethke, N., and Tricker, E. (2022) Predictably Unpredictable: A Look at Autocorrelation in Market Variables. Graham Capital Management.

[1] -> Why CNN could be a boon to predict prices \
[2] -> ResNet paper, and how to circumvent vanishing gradient with identity and projection shortcuts  \
[3] -> Inception v4 architecure which introduce shortcuts form [2], and display better results then pure ResNet \
[4] -> Vol is autocorrelated (realized vol, be careful with microstructure if more HF, choose the estimator wisely) 

---

## Notes from the kickoff session

**One-liner.** Predict short-term (intraday) **realized volatility** with a **neural-network** approach, using market data and **news as features**.

---

## Goals
- Focus on **realized volatility (RV)** prediction (intraday / short term).
- You may choose a **cross-sectional (XS)** or **time-series (TS)** framing.
- Free to **choose the stock universe** (and compare types of stocks if useful).

## Data
- **No dataset is provided** — we must **source/build the data** ourselves.
- **News inputs** are explicitly in scope as predictive features.

## Approach & Scope
- Take a **neural-network perspective** (not restricted to a single architecture).
- This is **research-oriented**: exploration over a fixed recipe.

## Process & Support
- **Limited predefined guidance**; autonomy expected.
- **Weekly meetings**; ask questions as needed.
