# Architecture

## Visual Expert
EfficientNet-B0 + projection MLP -> 128-d L2-normalised embedding

## Text Expert
TrOCR-small-printed (encoder only) + layout histogram -> 128-d

## Signal Expert
30 SRM filters + FFT magnitude (31-channel) -> ResNet-like -> 128-d

## Judge
Log-odds accumulator: lambda_t += log(q/(1-q)); b_t = sigmoid(lambda_t)
Actions: {query_expert_0, query_expert_1, query_expert_2, declare}
Heads: probability + policy + value

## Episode Flow
1. Prior b_0=0.5 (lambda_0=0)
2. Judge queries expert i -> lambda updated -> b updated
3. MSR reward r_t = log b_t(y) - log b_{t-1}(y) given to expert
4. Judge declares (or forced at step 4)

## Stage 1: Metric Learning
InfoNCE + ContrastiveAlignment + Triplet loss
Objective: eR close to eT for genuine, far for forged
Enables zero-shot: model learns comparison not classification

## Stage 2: MAPPO
CTDE: centralised critic, decentralised execution
Alternating schedule: 5 epochs judge, 5 epochs experts, repeat
Decorrelation penalty prevents expert collapse
