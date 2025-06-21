# World Model
🧭 世界模型（World Models）发展时间线

⸻

🔹 2018｜David Ha & Schmidhuber 提出 “World Models” 💡

📄 论文：World Models
🔍 核心思想：
	•	使用 VAE（视觉编码）+ RNN（记忆）+ 控制器（策略）
	•	在 latent space 中模拟未来，从而实现 dream-based 训练
	•	可在模拟器生成的“梦境”中训练强化学习 Agent

📌 意义：

首次明确提出了“世界模型”的概念，并给出结构可复现、效果显著的 RL 框架。

⸻

🔹 2019｜Dreamer 系列开端：学习 latent 环境模型 🌙

📄 论文：Dream to Control: Learning Behaviors by Latent Imagination
作者：Danijar Hafner（DeepMind）

🔍 核心改进：
	•	使用 Recurrent State Space Model（RSSM） 代替 RNN
	•	利用 latent 预测训练策略，显著提升 sample efficiency

📌 意义：

世界模型正式步入「强化学习主力技术」阵营，成为 model-based RL 的核心手段。

⸻

🔹 2020–2021｜DreamerV2、PlaNet 等持续优化 🧠

📄 论文：
	•	DreamerV2
	•	PlaNet: Learning Latent Dynamics

🔍 改进方向：
	•	更稳定的 latent 表达方式（stochastic latent state）
	•	使用 image reconstruction + reward prediction 联合训练

📌 关键词：视觉控制、低数据训练、强化学习、世界模型

⸻

🔹 2022｜LeCun 提出 JEPA 🌌

📄 论文：Joint Embedding Predictive Architecture (JEPA)
提出者：Yann LeCun（Meta 首席科学家）

🔍 与 David Ha 的对比：

特点	David Ha World Model	JEPA
类型	生成 + 强化学习导向	自监督结构学习导向
输出	显式重构/生成（pixel）	latent space embedding
架构	VAE + RNN	编码器 + 预测器
模型用途	控制 Agent	感知 + 结构预测

📌 意义：

把世界模型从“控制智能体”拓展到“感知与世界理解”的普适表示学习范式。

⸻

🔹 2023–2024｜JEPA 融合 MAE、VICReg 等自监督表示学习 🔁

代表论文：
	•	MAE: Masked Autoencoders
	•	VICReg: Variance-Invariance-Covariance Regularization
	•	Facebook 发布 ImageNet-JEPA 实现

🔍 关键词：
	•	预测 embedding，而非生成像素
	•	不再依赖语言、token、生成任务
	•	可泛化到图像、视频、音频、语义结构

📌 研究趋势：

世界模型不再是 RL 专属，而是成为普适的结构感知机制，与大模型协同发展。

⸻

🔹 2024–2025｜World Models + LLM + 多模态 + 智能体集成 🤖

研究热点（截至目前）：
	•	用世界模型增强多智能体规划（如 CAMEL、Voyager 等 Agent 框架）
	•	利用世界模型生成模拟场景，供 LLM fine-tune 或推理
	•	与 memory systems、反事实推理、智能体长期预测结合

📌 代表性观点：

LeCun：“LLM 是边角料预测机器，真正的智能需要世界模型”
Hafner：“Latent imagination 是通往 AGI 的路径之一”

⸻

📌 总结：发展关键路径

年份	事件	代表人物	模型
2018	World Models 概念首次提出	David Ha	VAE + RNN
2019	Dreamer 提高训练效率	Hafner	RSSM
2021	DreamerV2、PlaNet 等稳定演化	Hafner 等	Hybrid latent models
2022	LeCun 提出 JEPA，拓展至结构学习	Yann LeCun	编码器 + latent predictor
2023+	世界模型走向通用表征/感知系统	LeCun / Meta AI	多模态 + 非对比 + 动态预测