远程会议的录音一进后期，人声后面总拖着一层糊糊的尾巴：近讲人声明明干净，字尾却像被毛玻璃擦过，会议室那点混响一重叠，辅音先糊，齿音随后发毛。dereverb/suppress.py 里 stft_mag、decay_estimate、spectral_subtract、wiener_gain、smooth_gain、reconstruct、validate、dereverb_batch 八个函数现在都只抛 NotImplementedError，而验收测试锁定了十处最容易写错的隐蔽缺陷：decay_estimate 用当前帧最大值而不是回溯 k 帧的斜率，晚期包络偏高；spectral_subtract 不设下限，负幅度被截成 0，产生空洞；wiener_gain 的分母漏掉 late 项，增益恒接近 1；smooth_gain 用时间平滑但不跨频带，出现孤立尖峰；reconstruct 用幅度谱乘原相位重建，但 hop 不一致，长度变化；reconstruct 不做窗和归一，重叠区幅度起伏；validate 不检查 gain>1，过增益未被报；stft_mag 的窗不加周期修正，帧间能量不一致；dereverb_batch 复用晚期估计，第二条轨道用第一条的包络；smooth_gain 的 alpha 未钳制到 0..1，越界后增益发散。dereverb/suppress.py 是核心，tests/test_suppress.py 15个红。要修的是：按各函数 docstring 写明的契约逐项修复上述十处缺陷，让 15 个测试全部通过，并且不修改 tests。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始状态必须恰好是 `15 failed`，不能出现 collection error、import error 或 syntax error；15 个失败都应来自测试调用桩后触发的 `NotImplementedError`。修好实现后应得到 `15 passed`。整个题目离线完成，只用 Python 标准库 `math` 和 pytest，禁止随机数、网络与 sleep。幅度一律按线性值处理，只有直混比用 `20 * log10(幅度比)` 换算成 dB；数值比较默认容差 `1e-9`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_suppress.py` 是验收基准，不得修改、删除、跳过、xfail、改小容差或改弱断言。实现只能改 `dereverb/suppress.py`（必要时补 `dereverb/__init__.py` 的导出），不能靠改测试迁就实现。
