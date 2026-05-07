# Round 1 vs Round 2: depth-extrapolation comparison

- Round 1 ckpt: `/home/alexm/OpenMythos/checkpoints_3b_loops4_fast/step_0006103.pt` step 6103; trained at fixed T = 4
- Round 2 ckpt: `checkpoints_3b_varT_fast/step_0012207_full.pt` step 12207; trained at variable T = 4 max

## FineWeb-Edu held-out CE

| ACT | K | round 1 loss | round 1 ppl | round 2 loss | round 2 ppl | delta loss |
|-----|---|--------------|-------------|--------------|-------------|------------|
| off | 4 | 4.7602 | 116.77 | 4.2455 | 69.79 | -0.5147 |
| off | 8 | 6.2272 | 506.36 | 4.2522 | 70.26 | -1.9751 |
| off | 16 | 7.3069 | 1490.54 | 4.2525 | 70.28 | -3.0544 |
| off | 32 | 7.4563 | 1730.71 | 4.2524 | 70.28 | -3.2039 |
| on | 4 | 4.2213 | 68.12 | 4.1130 | 61.13 | -0.1083 |
| on | 8 | 4.2213 | 68.12 | 4.1130 | 61.13 | -0.1083 |
| on | 16 | 4.2213 | 68.12 | 4.1130 | 61.13 | -0.1083 |
| on | 32 | 4.2213 | 68.12 | 4.1130 | 61.13 | -0.1083 |

## Round 2 GSM8K answer-only CE

| K | answer-only loss | ppl | tokens measured |
|---|------------------|-----|-----------------|
| 4 | 4.1348 | 62.48 | 5172 |
| 8 | 4.1348 | 62.48 | 5172 |
| 16 | 4.1348 | 62.48 | 5172 |
| 32 | 4.1348 | 62.48 | 5172 |

## Round 2 TinyStories CE (general-distribution sanity probe)

| K | loss | ppl | tokens measured |
|---|------|-----|-----------------|
| 4 | 3.8242 | 45.80 | 65536 |
| 8 | 3.8242 | 45.80 | 65536 |
| 16 | 3.8242 | 45.80 | 65536 |
| 32 | 3.8242 | 45.80 | 65536 |

## Round 2 generation samples (vs round 1 samples in `gen_samples_round1.txt`)

Prompt: `The recurrent-depth transformer architecture`

```
K=4: The recurrent-depth transformer architecture, a new kind of "pension" of the 16th century, is now often referred to as the "pension" of a "pension" of the 17th century.
K=8: The recurrent-depth transformer architecture, a new kind of "pension" of the 16th century, is now often referred to as the "pension" of a "pension" of the 17th century.
K=16: The recurrent-depth transformer architecture, a new kind of "pension" of the 16th century, is now often referred to as the "pension" of a "pension" of the 17th century.
K=32: The recurrent-depth transformer architecture, a new kind of "pension" of the 16th century, is now often referred to as the "pension" of a "pension" of the 17th century.
```

Round 1 samples for the same prompt:

```
K=4: The recurrent-depth transformer architecture, a new kind of architecture, which, according to the Institute of Technology, is now under development. The main idea behind this new architecture is that it is actually very important to develop a modern architecture
K=8: The recurrent-depth transformer architecture, a new kind of architecture, which, according to the Institute of Technology, is now under development. The main idea behind this new architecture is that it is actually very important to develop a modern architecture
K=16: The recurrent-depth transformer architecture, a new kind of architecture, which, according to the Institute of Technology, is now under development. The main idea behind this new architecture is that it is actually very important to develop a modern architecture
K=32: The recurrent-depth transformer architecture, a new kind of architecture, which, according to the Institute of Technology, is now under development. The main idea behind this new architecture is that it is actually very important to develop a modern architecture
```
