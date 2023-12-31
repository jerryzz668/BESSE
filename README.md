# Be Everyone's Seq2Seq Engine
---
本项目提供一个通用的基于Bert的EncoderDecoderModel模型，实现seq2seq。该项目目前仅支持单GPU训练。

## 数据集
两个文件，可以理解为对联，一个上联，一个下联。如：

train_sources:
```
晚 风 摇 树 树 还 挺 
愿 景 天 成 无 墨 迹 
丹 枫 江 冷 人 初 去 
忽 忽 几 晨 昏 ， 离 别 间 之 ， 疾 病 间 之 ， 不 及 终 年 同 静 好 
闲 来 野 钓 人 稀 处 
```

train_targets:
```
晨 露 润 花 花 更 红 
万 方 乐 奏 有 于 阗 
绿 柳 堤 新 燕 复 来 
茕 茕 小 儿 女 ， 孱 羸 若 此 ， 娇 憨 若 此 ， 更 烦 二 老 费 精 神 
兴 起 高 歌 酒 醉 中 
```

## 环境准备
下载bert-base-chinese
```bash
pip install -U huggingface_hub
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --resume-download --local-dir-use-symlinks False bert-base-chinese --local-dir bert-base-chinese
```
conda环境
```bash
pip install -r requirements.txt
```

## Train
尽可能简化的一个train代码实现。

## Infer
infer