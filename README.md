# 冷却盘管选型核算服务

把冷却盘管选型时照焓湿图手描装置露点（ADP）、凑旁通系数（BF）的迭代活儿
钉死成一个常驻 HTTP 服务。给定进口状态，再给已知旁通系数或目标出风，
服务反算装置露点、旁通系数、出口全部状态量与冷量，谁都能拿它核对手算。

- Python 3.12 + FastAPI，无持久化，每次请求独立计算；
- 饱和蒸汽压统一使用 **固定的 Magnus 公式**（ASHRAE Fundamentals 系数），
  全项目只有 `app/psychrometrics.py` 一处公式来源，不在别的模块另抄；
- 含湿量、焓、饱和蒸汽压集中一处；ADP 迭代、旁通加权、显热/潜热分解各自
  独立成模块；
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
  schemas.py         HTTP 请求/响应 Pydantic 模型
  main.py            FastAPI 路由与统一结构化错误
  errors.py          领域错误类型与错误码
tests/               随仓自动化测试（63 个用例）
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

错误响应统一信封：

```json
{"error": {"code": "not_dehumidifying", "message": "装置露点 … 不低于进口湿球温度 …"}}
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

范围仅限冷却盘管选型核算，只经 HTTP 提供计算能力，无图形界面。
