# AGENTS.md

## Responses

Be brief. Explain only what is necessary.

## Before Writing Code

Identify the contract, data flow, and smallest coherent implementation.

Prefer clear, serial execution unless another structure is required.

## Coding Style

* Prefer explicit code over cleverness or code golf.
* Keep execution order visible.
* Validate at system boundaries, then trust internal code.
* Do not add speculative fallbacks or unnecessary `try/except` blocks.
* Fail loudly with useful errors.

## Design

* Give each function one clear purpose.
* Pipeline functions may coordinate smaller functions in a defined sequence.
* Do not fragment obvious sequential logic into unnecessary helpers.
* Use explicit inputs and return values.
* Avoid hidden state and undeclared side effects.
* Define clear types or schemas for shared data.
* Keep configuration and domain data in one source of truth.
* Add abstractions only when they reduce real complexity.

## Consistency

* Follow existing naming, structure, typing, and formatting.
* Comment only on invariants, constraints, and non-obvious decisions.
* Add dependencies only when they provide clear value.
* Test core behavior, boundaries, and known failures—not hypothetical edge cases.
