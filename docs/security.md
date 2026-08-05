# Security

## Treat model artifacts as code

Python pickle, joblib and some framework serialization formats can execute code while loading.
Only load artifacts produced by a trusted build pipeline or obtained from a source whose
integrity you have verified.

FlexPredict reduces accidental ambiguity but cannot make an unsafe serialization format safe:

- PyTorch `.pt` and `.pth` files are never guessed automatically;
- state dictionaries default to `torch.load(..., weights_only=True)`;
- loading a complete PyTorch model requires `loader="torch_model"` and defaults to
  `weights_only=False`;
- ensemble weight `.npy` files use `np.load(..., allow_pickle=False)`.

## Validate application input

Use an explicit {class}`~flexpredict.InputSchema` at external trust boundaries. It can reject
extra fields, enforce required and nullable values, coerce types and run domain validators
before inference.

Validators are application code. Keep them deterministic, fast and free of side effects.
Avoid including sensitive values in custom exception messages that may reach logs.

## Resource limits

FlexPredict validates shapes and empty batches, but it does not impose maximum batch sizes,
request sizes, inference timeouts or memory limits. Enforce those limits in the service or job
runner around the library. A custom remote engine should implement its own network timeout.

## Output validation

Numeric results must be real and finite; probabilities must be in `[0, 1]`; label objects are
restricted to strings, booleans and finite real scalars. These checks prevent malformed model
output from silently propagating, but they do not establish model accuracy, calibration or
fairness.

Report suspected vulnerabilities privately to the project maintainer rather than publishing
an exploit in a public issue.
