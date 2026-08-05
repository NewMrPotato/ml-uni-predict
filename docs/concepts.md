# Core concepts

## The inference pipeline

Every {class}`~flexpredict.Predictor` follows the same sequence:

1. interpret the input as one sample or a batch;
2. select, order and validate features when a schema is present;
3. apply an optional preprocessor;
4. execute the selected inference engine;
5. select one native model output when necessary;
6. normalize the output to `(n_samples, n_outputs)`;
7. validate its task and output semantics;
8. return a {class}`~flexpredict.PredictionResult`.

This separation lets each ensemble member keep its own schema, preprocessing pipeline,
framework and device while sharing an application-level input and output contract.

## Feature shortcut versus explicit schema

`features=[...]` creates a selection schema with `dtype=None` and `extra_fields="ignore"`.
It is ideal when feature presence and order matter but validation does not.

An explicit {class}`~flexpredict.InputSchema` defaults to coercion and
`extra_fields="forbid"`. Use it at trust boundaries or when input errors should be reported
before the model is called.

You cannot pass both `features` and `schema`.

## Task and output kind

FlexPredict attempts to infer classification or regression metadata from modern sklearn
tags, then falls back to `_estimator_type`. It also reads `classes_` without importing
scikit-learn. For framework-native and custom models, set `task` explicitly when the
semantics matter.

| Task | Typical output kind | Meaning |
| --- | --- | --- |
| `regression` | `values` | finite real predictions |
| `classification` | `labels` | strings, booleans or finite real labels |
| `classification` | `probabilities` | finite real values in `[0, 1]` |
| `classification` | `logits` | finite real, unnormalized class scores |
| `unknown` | `values` | generic finite real output |

Regression predictors cannot be configured with labels, probabilities or logits.
Probability range validation does not require rows to sum to one; calibration and class
normalization remain the model's responsibility.

## Shape normalization

Native model outputs may be scalar, one-dimensional or two-dimensional. FlexPredict applies
these rules:

| Native output | Interpretation |
| --- | --- |
| scalar | one output for one input sample only |
| 1D, length equals batch size | one output per sample |
| 1D, one input sample | multiple outputs for that sample |
| 2D | already `(n_samples, n_outputs)` |

Higher-dimensional outputs and mismatched sample counts are rejected. Use `output_selector`
or a custom engine when the native result needs another interpretation.

## Predictor versus ensemble

A predictor owns one input-to-result pipeline. An {class}`~flexpredict.EnsemblePredictor`
calls multiple predictor-like objects with the original input, verifies that their
{class}`~flexpredict.PredictionResult` objects are compatible, aligns probability or logit
columns by class metadata when possible, and aggregates the stacked values.

The ensemble deliberately does not silently combine different tasks, output kinds, batch
shapes or incompatible class sets.
