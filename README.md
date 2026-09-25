# 冷却盘管选型核算服务

把冷却盘管选型时照焓湿图手描装置露点（ADP）、凑旁通系数（BF）的迭代活儿
钉死成一个常驻 HTTP 服务。给定进口状态，再给已知旁通系数或目标出风，
服务反算装置露点、旁通系数、出口全部状态量与冷量，谁都能拿它核对手算。
真实机房里的多级处理链（预冷 → 深度除湿 → 再热/调温，几级串在一根风道上）
也能作为**一个整体**一次算完：级间状态传递、全链守恒收口、全链目标反推
都由服务自己钉死，不用把每一级的出口手抄下来当下一级的进口。

- Python 3.12 + FastAPI，无持久化，每次请求独立计算；
- 饱和蒸汽压统一使用 **固定的 Magnus 公式**（ASHRAE Fundamentals 系数），
  全项目只有 `app/psychrometrics.py` 一处公式来源，不在别的模块另抄；
- 含湿量、焓、饱和蒸汽压集中一处；ADP 迭代、旁通加权、显热/潜热分解各自
  独立成模块；多级链只是在单级能力之上的一层耦合，不复制单级逻辑；
- 出口含湿量与焓必须用**同一个 BF 线性加权**，出风温度由加权后的
  `(h_out, W_out)` 反解，绝不单独做温度算术平均。

## 目录结构

```
app/
  psychrometrics.py  湿空气基础公式（全项目唯一出处）：p_ws/W/h/露点/湿球
  states.py          状态展开：RH/露点/含湿量三选一 → 统一 AirState
  numerics.py        ADP 反算用的二分与号变扫描
  adp.py             装置露点求解 + 旁通加权（五种可解组合）
  cooling.py         总冷量/显热/潜热/SHR 分解
  chain.py           多级串联链：级间传递、全链守恒收口、全链目标反推
  schemas.py         HTTP 请求/响应 Pydantic 模型
  main.py            FastAPI 路由与统一结构化错误
  errors.py          领域错误类型与错误码
tests/               随仓自动化测试（115 个用例）
```

## 构建与运行

```bash
docker build -t cooling-coil .
docker run --rm -p 8000:8000 cooling-coil
```

镜像构建过程中会先跑全套 pytest，**测试不过镜像不产出**。

本地直接跑：

```bash
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
pytest            # 跑测试
```

服务起来后：

- Swagger UI：<http://localhost:8000/docs>
- 健康检查：`GET /`
- 示范工况：`GET /demo`（见文末）

## 物理约定

| 量 | 单位 | 说明 |
|---|---|---|
| 温度 | °C | 干球 t_db、露点 t_dp、湿球 t_wb、装置露点 t_adp |
| 压力 | Pa | 大气压 P，缺省标准大气压 101325 |
| 含湿量 W | kg(水)/kg(干空气) | 另给 `w_g_per_kg`（g/kg）便于核对 |
| 焓 h | kJ/kg(干空气) | `h = 1.006·t + W·(2501 + 1.86·t)` |
| 冷量 | kW | 以干空气质量流量 `m_da`（kg/s）为基准 |

饱和蒸汽压（Magnus，系数固定写死在 `app/psychrometrics.py`）：

```
ln(p_ws) = c1/T + c2 + c3·T + c4·T² + c5·T³ + c6·ln(T),  T = t + 273.15
```

旁通混合（**W 与 h 用同一个 BF**）：

```
W_out = W_adp + BF·(W_in − W_adp)
h_out = h_adp + BF·(h_in − h_adp)
t_out 由 (h_out, W_out) 反解
```

冷量分解（对同一组进出口状态自洽）：

```
Q_total    = m_da·(h_in − h_out)
Q_sensible = m_da·(1.006 + 1.86·W_out)·(t_in − t_out)
Q_latent   = Q_total − Q_sensible
SHR        = Q_sensible / Q_total  ∈ (0, 1]
```

## 边界条件（违反即结构化 422，绝不硬凑一个数）

- 含湿量超过该温度气压下的饱和上限（反算 RH > 1）→ `invalid_state`；
- 装置露点必须**严格低于进口湿球温度**，且 ADP 含湿量不得超过进口含湿量
  （否则不是冷却去湿工况，例如出风比进口还湿）→ `not_dehumidifying`；
- 旁通系数 BF ∈ [0, 1] → 否则 `invalid_bypass_factor`；
- SHR ∈ (0, 1] → 否则 `invalid_shr`；
- ADP 迭代不收敛、或过程线延伸到温度下限仍不与饱和曲线相交 →
  `no_convergence` / `inconsistent_state`，**绝不上一次请求的中间值**；
- 湿度三种表示必须恰好给一种；请求字段用 `extra="forbid"` 拒绝拼错的字段。

多级链追加的边界（错误信封里多带一个 `stage_index` 定位到第几级）：

- 空链、级已知量组合冲突、级间气压不一致 → `invalid_request`，
  定位到具体级次；
- 某一级的设定让该级出口（即下一级进口）落到饱和线外侧、或该级 ADP
  高于该级进口湿球 → `not_dehumidifying`，定位到该级，绝不把物理上
  讲不通的状态往下传；
- 沿链温度回升、含湿量沿链增加 → `not_dehumidifying`（纯旁通透传级
  进出口一致，天然合法）；整条链没有任何一级做功 → 无效工况；
- 全链目标反推时：自由量多于方程数 → `chain_underdetermined`；
  各级约束与全链目标彼此矛盾 → `chain_overdetermined`；
  阻尼牛顿迭代不收敛 → `no_convergence` 并指明卡在第几级，
  中途试探的临时装置露点不作为结果返回；
- 收口处校验：全链总冷量必须等于各级冷量之和、总去湿量必须等于各级
  去湿量之和（望远镜守恒，数值容差内），不自洽 → `inconsistent_state`。

错误响应统一信封（`stage_index` 仅多级链场景非空）：

```json
{"error": {"code": "not_dehumidifying", "message": "第 2 级：装置露点 … 不低于进口湿球温度 …", "stage_index": 2}}
```

## 接口

### 1. `POST /coil` —— 盘管核算（ADP + 出口状态 + 冷量）

请求体：`inlet`（必填）+ 下列任意一组已知量 + 可选 `m_da`（默认 1 kg/s）。

| 模式 | 要给的字段 | 说明 |
|---|---|---|
| 正算 | `t_adp_c` + `bf` | 直接加权出风 |
| 反算 1 | `outlet`（完整出风状态） | 过程线与饱和曲线求交得 ADP，再反解 BF |
| 反算 2 | `target_t_out_c` + `target_shr` | 先由 SHR 闭式解出 W_out，再走反算 1 |
| 反算 3 | `bf` + `target_t_out_c` | 二分 ADP 使出风温度命中目标 |
| 反算 4 | `bf` + `target_shr` | 二分 ADP 使 SHR 命中目标 |

**条件不够就不算**：只给目标出风温度、只给目标 SHR、或只给 BF 都是欠定
（两个混合方程含两个未知量），返回 `invalid_request`，不猜解；
正算只给 ADP 温度不给 BF 同样拒绝。已知量互相冲突（例如正算参数与
目标参数同给）也拒绝。

状态对象里湿度三选一：`rh`（0~1）、`t_dp_c`、`w`。

```bash
curl -s http://localhost:8000/coil -H 'Content-Type: application/json' -d '{
  "inlet": {"t_db_c": 35.0, "p_pa": 101325, "rh": 0.5},
  "t_adp_c": 7.0, "bf": 0.2
}'
```

反算示例（目标出风温度 + 目标显热比）：

```bash
curl -s http://localhost:8000/coil -H 'Content-Type: application/json' -d '{
  "inlet": {"t_db_c": 35.0, "rh": 0.5},
  "target_t_out_c": 20.0, "target_shr": 0.60
}'
```

### 2. `POST /state` —— 只展开湿空气状态

```bash
curl -s http://localhost:8000/state -H 'Content-Type: application/json' -d \
  '{"t_db_c": 35.0, "rh": 0.5}'
```

返回统一含湿量 W、RH、露点、湿球、焓。用 `rh` 或 `t_dp_c` 描述同一状态，
展开出的 W 在 1e-10 容差内一致，下游盘管结果不跳变（有测试钉住）。

### 3. `POST /loads` —— 只核对报告数字（不做 ADP 迭代）

```bash
curl -s http://localhost:8000/loads -H 'Content-Type: application/json' -d '{
  "inlet":  {"t_db_c": 35.0, "rh": 0.5},
  "outlet": {"t_db_c": 12.7, "w": 0.00852},
  "m_da": 1.0
}'
```

返回总冷量、显热、潜热与 SHR，用来核对盘管选型报告里报的数字。

### 4. `POST /chain` —— 多级串联盘管链（一次算完整条链）

一条链 = 共用进口 `inlet` + 按风道顺序排列的 `stages` + 可选全链目标
`target` + 可选 `m_da`。前一级的出口原样成为后一级的进口，最后一级的
出口就是整条链的出风。每一级的已知量给法与 `/coil` 完全一致、可以混用：

```bash
curl -s http://localhost:8000/chain -H 'Content-Type: application/json' -d '{
  "inlet": {"t_db_c": 35.0, "rh": 0.5},
  "stages": [
    {"name": "预冷",   "t_adp_c": 10.0, "bf": 0.5},
    {"name": "深除湿", "target_t_out_c": 14.0, "target_shr": 0.55},
    {"name": "调温",   "bf": 0.3, "target_t_out_c": 12.0}
  ],
  "m_da": 2.0
}'
```

响应逐级给出进口、装置露点、旁通系数、出口、本级冷量分摊
（`load_share` = 本级冷量 / 全链总冷量）与是否透传级（`passthrough`），
末尾给全链合计：总冷量、总显热、总潜热、总体显热比、总去湿量。

**带全链目标反推**：`target` 给最终目标出风状态（`outlet`）或
`target_t_out_c` + `target_shr`，各级里没给足的级（只给 `bf`、只给
`t_adp_c`、只给半个目标、或什么都不给）由链级反推解出装置露点/旁通系数，
使整条链恰好落到目标：

```bash
curl -s http://localhost:8000/chain -H 'Content-Type: application/json' -d '{
  "inlet": {"t_db_c": 35.0, "rh": 0.5},
  "stages": [
    {"t_adp_c": 10.0, "bf": 0.5},
    {},
    {"t_adp_c": 6.5, "bf": 0.5}
  ],
  "target": {"target_t_out_c": 10.94, "target_shr": 0.491}
}'
```

链级求解方式（响应里的 `mode`）：

| mode | 含义 |
|---|---|
| `forward` | 无全链目标，各级已知量给全，逐级正算 |
| `verify` | 各级给全 + 全链目标，正算后核验目标一致性 |
| `inverse` | 带全链目标反推，解出未给足级的 ADP/BF |

自由量数与方程数对不上时拒绝：欠定 `chain_underdetermined`、
过定 `chain_overdetermined`、迭代不收敛 `no_convergence`（带级次）。
若同一组约束存在多个孤立解（过程线与饱和曲线可有两个交点，与单级
完整出风反算同理），按与单级一致的确定性约定取装置露点最暖的解。

## 示范工况（`GET /demo`）

夏天常见工况：35 °C、50%RH、常压进风，ADP 7 °C，BF = 0.2。
手算量级：

| 量 | 进口 | ADP（饱和） | 出口 |
|---|---|---|---|
| 干球温度 °C | 35.0 | 7.0 | ≈ 12.7 |
| 含湿量 g/kg | ≈ 17.8 | ≈ 6.2 | ≈ 8.5 |
| 焓 kJ/kg | ≈ 80.8 | ≈ 22.7 | ≈ 34.3 |

单位干空气总冷量 ≈ 46.5 kJ/kg，SHR ≈ 0.49（显热 ≈ 潜热）。

## 自动化测试钉住的关系

- BF = 0：出口精确等于 ADP 饱和状态（温度/含湿量/焓，1e-10 量级）；
- BF = 1：出口等于进口、总/显热/潜热冷量全为零；
- 固定 ADP，BF 0.1 → 0.3：出口 W、h 更靠近进口，冷量单调下降；
- W 与 h 由同一个 BF 加权（且出风温度确实不是温度算术平均）；
- 显热 + 潜热 = 总冷量（容差内），质量流量线性缩放冷量但不改变 SHR；
- RH 入口与露点入口展开一致、下游出口结果不跳变；
- 五种求解模式相互自洽（正算结果喂给反算能还原 ADP/BF）；
- 超饱和、BF/SHR 越界、ADP 高于湿球/比进口湿、欠定输入、冲突输入、
  物理不可达目标、不自洽进出口对——各类非法输入均被结构化拒绝；
- 请求间无状态串台。

多级链（`tests/test_chain.py`、`tests/test_chain_http.py`）：

- 一级链的每一项与 `/coil` 对同样输入的回答完全一致（浮点量级）；
- 两级链拆开用单级核算按「前级出口即后级进口」手工串一遍，每一级与
  最终出口都吻合；级间传递的是同一个状态对象，逐位一致；
- 全链总冷量 == 各级冷量之和、总去湿量 == 各级去湿量之和；
- 沿链温度不回升、含湿量不增加；份额归一；
- 纯旁通透传级（BF=1）放行且冷量份额为零；全透传链判无效工况；
- ADP 高于该级进口湿球、级出口落入饱和线外侧、出风比进风还湿、
  级间气压不一致、空链——都返回定位到具体级次的结构化错误；
- 全链目标反推：正算结果作为目标喂回去，能还原各级 ADP/BF
  （自由级、两级各给一半、目标温度+SHR 等组合）；欠定、过定、
  不收敛分别结构化拒绝并指明级次。

范围仅限冷却盘管选型核算，只经 HTTP 提供计算能力，无图形界面。
