# Ver2.9.5 Path X 30k CER/WER 弱项诊断（γ.1）

日期：2026-07-22

## 结论先行

1. 主表 `ZH CER 12.00% / EN WER 8.25%` 来自外部严格评测（ZH1194 / EN567），不是内部 Full320。两层数据必须分开解释。
2. 外部严格评测中，高错误 case 并非主要由 >15 秒长音频构成；更强的直接信号是少数高错误尾部与内容错误拉高均值。
3. 内部 no_text 的跨语言 reference voice 并未系统性恶化内容错误；内容来自 source，ref language/cell 更影响音色转换而非应读文本。
4. 最强根因仍是内容通路/对齐不足：历史训练诊断显示 BNF cross-attention 近均匀、guided loss 高、phoneme accuracy 低；当前逐 case 结果与漏读/错读长尾一致。
5. 单纯继续训练不是首选。30k 相对 28k 有改善，但 20k→30k no_text CER 曾从 6.05% 变差至 6.78%，且 SIM/训练标量趋于平台。

## 口径与方法

- 外部严格集：ZH1194 用 Paraformer CER，EN567 用 Whisper-Large-v3 WER；这是论文主表口径。
- 内部 Full320：只把 160 条 `no_text` 作为 γ.1 主诊断样本；另用 160 条 `text` 作模式对照。内部 ASR 是 Qwen-ASR。
- `mean/std/median` 是 canonical per-case primary error 的宏平均。`micro*` 是本脚本按规范化字符/词重新计算 edit counts 后的微平均，仅用于切片比较，不替代官方汇总。
- 长度按 source audio：短 `<5s`、中 `5–15s`、长 `>15s`。
- 词汇丰富度：中文唯一规范化字符占比，英文唯一规范化词占比；在每种语言内按三分位分 low/medium/high。
- 尾静音 proxy：generated wav 末尾低于 `-40 dBFS` 的连续时长；`unfinished` 要求 duration ratio `<0.8` 且尾静音 `>=0.2s`。
- 重复 proxy：中文重复 bigram、英文重复 trigram 的超额占比；阈值 `>0.3`。内部同时沿用已有 repeat_score。
- `source_content` 只有 source text 与 target text 不同时才可判别。外部 strict VC 和内部 no_text 内容保持任务中二者相同，不能用这两组伪造“读 source 内容”结论。

## 一、外部严格评测：主表 12.00% / 8.25%

| 语言 | N | 指标 | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|---|
| ZH | 1194 | CER | 12.00% | 15.58% | 5.88% | 12.29% | 231 |
| EN | 567 | WER | 8.25% | 14.11% | 0.00% | 8.35% | 77 |

### ZH error 直方图

| 区间 | case 数 | 比例 |
|---|---|---|
| 0-5% | 525 | 43.97% |
| 5-10% | 213 | 17.84% |
| 10-20% | 217 | 18.17% |
| 20-50% | 185 | 15.49% |
| 50%+ | 54 | 4.52% |

### ZH 按 source 音频长度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| medium_5-15s | 829 | 13.17% | 16.73% | 7.41% | 13.33% | 179 |
| short_<5s | 365 | 9.36% | 12.18% | 5.56% | 9.47% | 52 |

### ZH 按内容词汇丰富度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| high | 390 | 10.36% | 13.56% | 5.56% | 10.56% | 59 |
| low | 417 | 13.49% | 16.59% | 7.41% | 13.85% | 102 |
| medium | 387 | 12.07% | 16.20% | 5.56% | 12.32% | 70 |

### ZH >20% 失败模式

- `primary error > 20%`：231 条。
- 模式允许多标签；下表先给互斥 primary mode，再给多标签计数。

Primary mode：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| omission | 12 | 5.19% |
| other_error | 176 | 76.19% |
| repetition | 37 | 16.02% |
| unfinished | 1 | 0.43% |
| wrong_content | 5 | 2.16% |

多标签计数：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| omission | 13 | 5.63% |
| other_error | 176 | 76.19% |
| repetition | 37 | 16.02% |
| unfinished | 1 | 0.43% |
| wrong_content | 5 | 2.16% |

Source-content 可判别状态： `{"not_identifiable_same_or_missing_source_text": 231}`。

### ZH Top 10 worst

1. `10003102-00000089_10003152-00000108` — CER `100.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: 传统吃法是白切，口感清淡，肉质滑嫩，原汁原味。
   - ASR: him
2. `00005237-00000052_00005238-00000086` — CER `100.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: 因为二克拉左右的主钻石周围，还有一圈粉钻簇拥。
   - ASR: 衣柜抛壳老鼠用在中物林他有个圈子正簇用中物林
3. `00004905-00000005_00004906-00000111` — CER `89.47%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: 但是总觉得从前这里，曾经有过骁勇强悍斗士。
   - ASR: 再次转缺个从前遮底成清整体从清勇过汹勇
4. `00004975-00000045_00004976-00000044` — CER `88.46%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`repetition`
   - Target: 小黑狗摆摆尾巴说，我身上没有汗腺，可舌头上有许多汗腺的呢。
   - ASR: 小黑狗要拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜拜
5. `00012910-00000493_00013044-00000102` — CER `86.36%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`omission`
   - Target: 谢师傅脑海中一片空白，半天才缓过神来，明白撞车啦。
   - ASR: 谢师傅
6. `00004774-00000013_00004775-00000061` — CER `86.36%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: 玉米苗长得有筷子那么高，鼹鼠妈妈又给玉米苗施肥。
   - ASR: 玉米苗施肥肥玉米苗施肥无疑很欢摇
7. `00004625-00000089_00004626-00000091` — CER `77.27%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: 周老师没有批评阿夸反而鼓励了他，这让阿夸很感动。
   - ASR: 乔瓦斯没有批评阿夸真人还夸韩我导购真人还夸韩我
8. `00005452-00000093_00005453-00000107` — CER `70.83%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`repetition`
   - Target: 党员干部勤政作风，不仅为群众做出了表率，也感动了外商。
   - ASR: 倒牙淡步勤政作风不显媚群众作风不显媚群众作风不
9. `00004549-00000042_00004550-00000011` — CER `70.37%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`repetition`
   - Target: 这名坚持在工作岗位上努力工作的医生，被评为最美儿科女医生。
   - ASR: 证明坚持在工作岗位上努力工作岗位上努力工作岗位上努力工作岗位上
10. `00013124-00000647_00013265-00000021` — CER `68.42%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`repetition`
   - Target: 导航开始，全程二十五公里，预计需要十二分钟。
   - ASR: 导航开始传承预计预计预计开始预计

### ZH Top 10 best

1. `00004501-00000032_00004504-00000108` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 全国恶性肿瘤发病，及死亡第一位的是肺癌。
   - ASR: 全国恶性肿瘤发病及死亡第一位的是肺癌
2. `00004509-00000108_00004510-00000018` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 妈妈从我手中拿走我歌曲的歌看，边看边赞不绝口。
   - ASR: 妈妈从我手中拿走我歌曲的歌看边看边赞不绝口
3. `00004510-00000116_00004511-00000014` — CER `0.00%`, lang=`zh`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: 十二个小时呀，简直无法想象怎么熬过来的。
   - ASR: 十二个小时呀简直无法想象怎么熬过来的
4. `00004511-00000098_00004512-00000028` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 将货物通关时间，从原来的九点二个小时，缩短为九分钟。
   - ASR: 将货物通关时间从原来的九点二个小时缩短为九分钟
5. `00004512-00000031_00004513-00000086` — CER `0.00%`, lang=`zh`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: 目前该案已抓获主要犯罪嫌疑人八人。
   - ASR: 目前该案已抓获主要犯罪嫌疑人八人
6. `00004513-00000013_00004514-00000036` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 另外，吃黄豆芽象征万事如意，吃发财芽寓意发财。
   - ASR: 另外吃黄豆芽象征万事如意吃发财芽寓意发财
7. `00004513-00000013_00004514-00000062` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 汇率是中国军事成本更低的，另一个不显眼的优势。
   - ASR: 汇率是中国军事成本更低的另一个不显眼的优势
8. `00004514-00000076_00004515-00000071` — CER `0.00%`, lang=`zh`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: 我还没完成周五阿顿分，给我的那个图表呢。
   - ASR: 我还没完成周五阿顿分给我的那个图表呢
9. `00004515-00000039_00004516-00000030` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 法院与不动产登记部门加强沟通，并督促银行提前办理抵押预约登记。
   - ASR: 法院与不动产登记部门加强沟通并督促银行提前办理抵押预约登记
10. `00004518-00000052_00004519-00000048` — CER `0.00%`, lang=`zh`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: 要不是彼得今天早上问起我，我也差不多都忘了。
   - ASR: 要不是彼得今天早上问起我我也差不多都忘了

### EN error 直方图

| 区间 | case 数 | 比例 |
|---|---|---|
| 0-5% | 328 | 57.85% |
| 5-10% | 75 | 13.23% |
| 10-20% | 84 | 14.81% |
| 20-50% | 61 | 10.76% |
| 50%+ | 19 | 3.35% |

### EN 按 source 音频长度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| medium_5-15s | 168 | 11.45% | 15.23% | 7.69% | 11.53% | 33 |
| short_<5s | 399 | 6.90% | 13.38% | 0.00% | 6.68% | 44 |

### EN 按内容词汇丰富度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| low | 199 | 8.92% | 14.93% | 0.00% | 9.13% | 27 |
| medium | 368 | 7.88% | 13.63% | 0.00% | 7.88% | 50 |

### EN >20% 失败模式

- `primary error > 20%`：77 条。
- 模式允许多标签；下表先给互斥 primary mode，再给多标签计数。

Primary mode：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| omission | 2 | 2.60% |
| other_error | 69 | 89.61% |
| repetition | 2 | 2.60% |
| wrong_content | 4 | 5.19% |

多标签计数：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| omission | 2 | 2.60% |
| other_error | 69 | 89.61% |
| repetition | 2 | 2.60% |
| wrong_content | 4 | 5.19% |

Source-content 可判别状态： `{"not_identifiable_same_or_missing_source_text": 77}`。

### EN Top 10 worst

1. `common_voice_en_18338543_common_voice_en_18466098-common_voice_en_18466096` — WER `100.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`wrong_content`
   - Target: Catch as catch can.
   - ASR: TechCam.
2. `common_voice_en_565900_common_voice_en_566283-common_voice_en_566282` — WER `86.67%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`wrong_content`
   - Target: A group of people are at a gathering where there is a large colorful flag
   - ASR: They might be keen for a phone, or at a gaver.
3. `common_voice_en_589496_common_voice_en_590785-common_voice_en_590791` — WER `83.33%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`wrong_content`
   - Target: Angela is pregnant, she told me.
   - ASR: Angelo's pregnant, shoot to Alderney.
4. `common_voice_en_30665077_common_voice_en_30735432-common_voice_en_30735433` — WER `75.00%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`wrong_content`
   - Target: The wooden shrine is generously proportioned for the three images it houses.
   - ASR: Through the within shrine, this is generously proportioned. This is generously proportioned.
5. `common_voice_en_647713_common_voice_en_648428-common_voice_en_648429` — WER `69.23%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`other_error`
   - Target: A young Arab, also loaded down with baggage, entered, and greeted the Englishman.
   - ASR: A young Arab entered and greeted the Amishmanth, intruded the Amishmanth,
6. `common_voice_en_21877403_common_voice_en_22364189-common_voice_en_22364186` — WER `61.54%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`repetition`
   - Target: As research continued, the protective effect of fluoride against dental decay was demonstrated.
   - ASR: As research continued, the protective, effective, effective, effective, effective, effective, effective, effective,
7. `common_voice_en_19854924_common_voice_en_19867710-common_voice_en_19867712` — WER `60.00%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`other_error`
   - Target: Whilst at college Dave, met singer and songwriter Adrian Snell.
   - ASR: Weilstock College, Dave, Adrian Snell, Adrian Snell.
8. `common_voice_en_19701583_common_voice_en_19706873-common_voice_en_19706870` — WER `60.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`other_error`
   - Target: He was offered a 'one year tour' playing with Alice.
   - ASR: It was offered a one-year tour, Langing with Kallus.
9. `common_voice_en_27794266_common_voice_en_27935469-common_voice_en_27935466` — WER `58.33%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`omission`
   - Target: The crater has nearly the same low albedo as the nearby surface.
   - ASR: the crater has nearby surface.
10. `common_voice_en_564578_common_voice_en_565900-common_voice_en_565901` — WER `57.14%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`other_error`
   - Target: He told them all to be seated.
   - ASR: We told them all, we ceded.

### EN Top 10 best

1. `common_voice_en_137148_common_voice_en_1416089-common_voice_en_1416090` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: Also, will numbers be written as digits or as words?
   - ASR: Also, will numbers be written as digits or as words?
2. `common_voice_en_15265_common_voice_en_153872-common_voice_en_153873` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: They set off running wildly into the trees.
   - ASR: They set off running wildly into the trees.
3. `common_voice_en_153872_common_voice_en_15734837-common_voice_en_15734838` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: Tyler, Lucy, Michelle, we're going to space!
   - ASR: Tyler, Lucy, Michelle, we're going to space.
4. `common_voice_en_15734837_common_voice_en_15903802-common_voice_en_15903807` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: Thousands of people die every year as a consequence of air pollution.
   - ASR: Thousands of people die every year as a consequence of air pollution.
5. `common_voice_en_15735839_common_voice_en_15903802-common_voice_en_15903815` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: History teaches us that humans do not learn from history.
   - ASR: History teaches us that humans do not learn from history.
6. `common_voice_en_15903802_common_voice_en_16666041-common_voice_en_16666040` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: I was well, but I'm all the better for being here.
   - ASR: I was well, but I'm all the better for being here.
7. `common_voice_en_16666041_common_voice_en_167249-common_voice_en_167247` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: Take these capsules over to Mrs. David's house.
   - ASR: Take these capsules over to Mrs. David's house.
8. `common_voice_en_167249_common_voice_en_17161-common_voice_en_17159` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: He must be disguised to avoid encounters with thieves.
   - ASR: He must be disguised to avoid encounters with thieves.
9. `common_voice_en_17147545_common_voice_en_17161-common_voice_en_17160` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`unknown`, mode=`below_20pct`
   - Target: The area was swirling in dust so intense that it hid the moon from view.
   - ASR: The area was swirling in dust so intense that it hid the moon from view.
10. `common_voice_en_17161_common_voice_en_17249419-common_voice_en_17249428` — WER `0.00%`, lang=`en`, length=`medium_5-15s`, cell=`unknown`, mode=`below_20pct`
   - Target: I heard the land where the hobbits live, the Shire, has actually been filmed in New Zealand.
   - ASR: I heard the land where the hobbits live, the Shire, has actually been filmed in New Zealand.

## 二、内部 Full320：no_text 160 条分层

内部 no_text 总体并不是主表 12%：ZH/EN 混合 primary error 宏平均为 `8.22%`。这里的价值是 cell、跨语言和部分 gender 元数据。


### 语言

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| en | 80 | 7.38% | 15.03% | 0.00% | 6.88% | 8 |
| zh | 80 | 9.05% | 15.40% | 0.00% | 8.55% | 14 |

### source 音频长度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| medium_5-15s | 40 | 8.36% | 14.18% | 0.00% | 7.73% | 6 |
| short_<5s | 120 | 8.17% | 15.58% | 0.00% | 7.88% | 16 |

### Source gender

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| female | 40 | 5.97% | 14.33% | 0.00% | 5.29% | 4 |
| male | 40 | 10.45% | 17.59% | 0.00% | 9.81% | 7 |
| unknown | 80 | 8.22% | 14.22% | 0.00% | 8.19% | 11 |

### Reference gender

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| female | 40 | 10.45% | 17.59% | 0.00% | 9.81% | 7 |
| male | 40 | 5.97% | 14.33% | 0.00% | 5.29% | 4 |
| unknown | 80 | 8.22% | 14.22% | 0.00% | 8.19% | 11 |

### Source/ref 是否跨语言

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| cross_language | 120 | 7.58% | 14.66% | 0.00% | 7.11% | 15 |
| same_language | 40 | 10.12% | 16.72% | 2.38% | 10.09% | 7 |

### 完整 cell

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| en_src_en_ref_same_gender | 20 | 8.55% | 12.22% | 3.57% | 9.44% | 3 |
| en_src_zh_ref_f2m | 20 | 6.88% | 17.52% | 0.00% | 5.62% | 2 |
| en_src_zh_ref_m2f | 20 | 10.98% | 19.74% | 0.00% | 9.66% | 3 |
| en_src_zh_ref_same_gender | 20 | 3.12% | 5.05% | 0.00% | 2.92% | 0 |
| zh_src_en_ref_f2m | 20 | 5.06% | 10.10% | 0.00% | 5.04% | 2 |
| zh_src_en_ref_m2f | 20 | 9.92% | 15.13% | 0.00% | 9.91% | 4 |
| zh_src_en_ref_same_gender | 20 | 9.54% | 13.77% | 0.00% | 8.78% | 4 |
| zh_src_zh_ref_same_gender | 20 | 11.68% | 20.12% | 2.38% | 10.54% | 4 |

### 内容词汇丰富度

| 组 | N | mean | std | median | micro* | >20% |
|---|---|---|---|---|---|---|
| low | 48 | 7.46% | 13.40% | 0.00% | 8.16% | 5 |
| medium | 112 | 8.54% | 15.96% | 0.00% | 7.72% | 17 |

### no_text error 直方图

| 区间 | case 数 | 比例 |
|---|---|---|
| 0-5% | 99 | 61.88% |
| 5-10% | 22 | 13.75% |
| 10-20% | 16 | 10.00% |
| 20-50% | 17 | 10.62% |
| 50%+ | 6 | 3.75% |

### no_text >20% 失败模式

- `primary error > 20%`：22 条。
- 模式允许多标签；下表先给互斥 primary mode，再给多标签计数。

Primary mode：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| other_error | 21 | 95.45% |
| wrong_content | 1 | 4.55% |

多标签计数：

| 模式 | case 数 | 占 >20% case |
|---|---|---|
| other_error | 21 | 95.45% |
| wrong_content | 1 | 4.55% |

Source-content 可判别状态： `{"not_identifiable_same_or_missing_source_text": 22}`。

### no_text Top 10 worst

1. `seedtts_no_text_en_src_zh_ref_f2m_000008` — WER `77.78%`, lang=`en`, length=`short_<5s`, cell=`en_src_zh_ref_f2m`, mode=`other_error`
   - Target: The hotel gives some complimentary water bottles to drink.
   - ASR: Don't tell. Give some to drink.
2. `seedtts_no_text_zh_src_zh_ref_same_gender_000005` — CER `75.00%`, lang=`zh`, length=`short_<5s`, cell=`zh_src_zh_ref_same_gender`, mode=`wrong_content`
   - Target: 充分发挥技术和经验的优势。
   - ASR: 充分發揮僅有的優勢。
3. `seedtts_no_text_en_src_zh_ref_m2f_000010` — WER `63.64%`, lang=`en`, length=`short_<5s`, cell=`en_src_zh_ref_m2f`, mode=`other_error`
   - Target: I can only repeat myself that this was not the plan.
   - ASR: I can't even repeat myself. That is what's multi-brand.
4. `seedtts_no_text_zh_src_zh_ref_same_gender_000002` — CER `56.25%`, lang=`zh`, length=`medium_5-15s`, cell=`zh_src_zh_ref_same_gender`, mode=`other_error`
   - Target: 滨海滨海贫民窟，公认是里约最危险的。
   - ASR: 滨海临滨海濒临库空人，比越最危险的。
5. `seedtts_no_text_en_src_zh_ref_m2f_000009` — WER `55.56%`, lang=`en`, length=`medium_5-15s`, cell=`en_src_zh_ref_m2f`, mode=`other_error`
   - Target: Like his predecessors, Cyrus had to recognize Median overlordship.
   - ASR: Like his predecessors, Median overlordship, overlordship.
6. `seedtts_no_text_en_src_zh_ref_m2f_000006` — WER `50.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_zh_ref_m2f`, mode=`other_error`
   - Target: Natalia is part of the San Antonio Metropolitan Statistical Area.
   - ASR: The Italian sports is San Antonio Metropolitan Statistical Area.
7. `seedtts_no_text_zh_src_en_ref_m2f_000008` — CER `47.06%`, lang=`zh`, length=`short_<5s`, cell=`zh_src_en_ref_m2f`, mode=`other_error`
   - Target: 副热带高压以下，简称副高已经向北移动。
   - ASR: 富州带高压一线，简称富冈，一定向北移动。
8. `seedtts_no_text_zh_src_en_ref_m2f_000005` — CER `46.67%`, lang=`zh`, length=`short_<5s`, cell=`zh_src_en_ref_m2f`, mode=`other_error`
   - Target: 优秀的产品经理要逐步变得更综合。
   - ASR: 优秀的产品经理要独孤不共通。
9. `seedtts_no_text_en_src_en_ref_same_gender_000018` — WER `41.67%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`other_error`
   - Target: See your email client to check your emails have been sent correctly
   - ASR: See your email client to check your emails have been. Check your emails have been.
10. `seedtts_no_text_zh_src_en_ref_same_gender_000007` — CER `41.18%`, lang=`zh`, length=`short_<5s`, cell=`zh_src_en_ref_same_gender`, mode=`other_error`
   - Target: 自动驾驶将为现有的司机运力提供补充。
   - ASR: 只供驾驶、交零信用的司机用，力提供补充。

### no_text Top 10 best

1. `seedtts_no_text_en_src_en_ref_same_gender_000000` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: She then joined her former colleagues to work on algorithmic financial market predictions.
   - ASR: She then joined her former colleagues to work on algorithmic financial market predictions.
2. `seedtts_no_text_en_src_en_ref_same_gender_000001` — WER `0.00%`, lang=`en`, length=`medium_5-15s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: It featured critical articles as well as reviews of short fiction and novels.
   - ASR: It featured critical articles as well as reviews of short fiction and novels.
3. `seedtts_no_text_en_src_en_ref_same_gender_000006` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: She regularly did judo to maintain her strength and flexibility.
   - ASR: She regularly did judo to maintain her strength and flexibility.
4. `seedtts_no_text_en_src_en_ref_same_gender_000007` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: "Victoria" will be used for training purposes until repairs are effected.
   - ASR: Victoria will be used for training purposes until repairs are effected.
5. `seedtts_no_text_en_src_en_ref_same_gender_000009` — WER `0.00%`, lang=`en`, length=`medium_5-15s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: Brooks advocates "growing" software organically through incremental development.
   - ASR: Brooks advocates growing software organically through incremental development.
6. `seedtts_no_text_en_src_en_ref_same_gender_000010` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: This implied accommodation close to the armed forces' Central War Room.
   - ASR: This implied accommodation close to the Armed Forces Central War Room.
7. `seedtts_no_text_en_src_en_ref_same_gender_000011` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: However, horses that could not be controlled had to be destroyed.
   - ASR: However, horses that could not be controlled had to be destroyed.
8. `seedtts_no_text_en_src_en_ref_same_gender_000013` — WER `0.00%`, lang=`en`, length=`medium_5-15s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: High gain is also used to induce audio feedback, which increases sustain dramatically.
   - ASR: High gain is also used to induce audio feedback, which increases sustain dramatically.
9. `seedtts_no_text_en_src_en_ref_same_gender_000016` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: A club acquaintance, and a mere one at that.
   - ASR: A club acquaintance, and a mere one at that.
10. `seedtts_no_text_en_src_en_ref_same_gender_000019` — WER `0.00%`, lang=`en`, length=`short_<5s`, cell=`en_src_en_ref_same_gender`, mode=`below_20pct`
   - Target: He chairs the executive committee of council and the community policing committee.
   - ASR: He chairs the Executive Committee of Council and the Community Policing Committee.

## 三、text mode 对照

| 模式 | N | mean primary error | std | >20% |
|---|---|---|---|---|
| no_text | 160 | 8.22% | 15.24% | 22 |
| text | 160 | 4.43% | 13.59% | 11 |

Text mode 明显更好，说明显式文本可绕过/补强弱内容对齐；但它改变了产品设定，不能直接拿来替代 no_text VC 主结果。它支持把 text-mode 比例实验作为 γ.2 的一个独立 probe，而不是直接改主表口径。

## 四、根因排序

### 1. F — BNF content cross-attention / 对齐强度不足（高置信）

- 直接现象：错误集中于读错、漏读及其他内容长尾；text mode 对照更好。
- 历史训练证据：content cross-attention normalized entropy `≈0.9991`、guided attention loss `≈0.980`、peak probability 仅 `≈1.33× uniform`，content injection 前置集中；effective-14k 附近 phoneme accuracy 约 `11.9%`。
- 这是当前最能同时解释“SIM 已强、CER/WER 落后”的根因。

### 2. D — 内容数据覆盖/难例分布不足（中等置信）

- 外部严格集显著差于内部 Full320，说明内部抽样低估真实长尾。
- 高错误 case 与复杂度/具体语句类型存在长尾，值得做 hard-case reweight 与数据补齐；但仅凭评测集不能证明训练数据具体缺哪类。

### 3. E — 训练不充分（低到中等置信，不能盲目续训）

- 支持面：内部 28k→30k no_text CER `7.81%→6.78%`，text `5.13%→3.90%`。
- 反证：20k→30k no_text CER `6.05%→6.78%`，SIM(ref) `0.4496→0.4485`，训练标量趋于平台。纯 30k→60k 不是首选，只适合作为 warm-restart 严格 stop-gated probe。

### 4. A — 长句问题（低置信）

- 当前 source 音频大多低于 15 秒，>15 秒样本不足以成为 12% 主因。应关注 duration mismatch/尾部失败，而不是笼统补长句。

### 5. B — 跨语言 reference 问题（低置信）

- 内部 no_text 中 source/ref 跨语言 cell 没有形成一致的内容错误劣化。外部严格集缺 cell 元数据，无法对主表做同样切片。

### 6. C — Speaker 泄漏导致读 source 内容（当前 no_text 不可验证）

- no_text VC 的 source text 本来就是 target text，因此“读 source 内容”在任务定义上不可判别。只有 text mode 的 counterfactual source/target 不同时可测，不应把它当成 12% CER 主因。

## 五、γ.2 针对性建议（仅建议，尚未授权执行）

1. **优先 Probe C：内容通路加强。** Adapter Conformer 2→4，guided attention `0.05→0.10`，phoneme classifier `0.02→0.05`；其余保持不变。这与第一根因直接对应。
2. **并行 Probe A-lite：hard-case reweight/补数据。** 不先笼统扩到 400k；先按本报告的高错误长度、复杂度和失败模式构造 50k 左右定向增量，并保持独立数据审计。
3. **Probe D 作为产品分支。** no_text:text 从 `1:0.3→1:1`，同时分别报告 no_text 与 text，不能用 text 数字替换 no_text 主表。
4. **训练规模化 Probe B 排在上述之后。** 只做 30k→33k 的短 warm-restart probe 和硬止损；先证明 CER 至少降 2pp，再授权 45k/60k。

建议 γ.2 首轮组合：`C`、`A-lite+C`、`D`。暂不把纯 `B` 列为前三。

## 六、限制

- 外部 strict 数据缺 gender/cell，因此 gender/cross-language 结论只来自内部 160 条 no_text。
- `same_gender` cell 没有绝对男女标签，source/ref gender 均记为 unknown；不进行猜测。
- 失败模式是可复现自动 proxy，不等于人工听审。Top worst 建议在 γ.2 设计前人工听 10 条确认。
- 外部 source text 与 target text 相同，source-content 泄漏不可判别。
