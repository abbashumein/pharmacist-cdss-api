# Project rules (I follow these on every change)

1. Every bug fix gets a regression test that FAILS before the fix and PASSES after.
2. No number in the README unless a script in the repo produces it from committed data.
3. Errors, quota failures and fallbacks are never counted as pass or fail. They are INVALID.
4. No secrets in code. Keys come from environment variables.
5. CI runs the tests on every push. A red build blocks the merge.
6. "Done" means: tests pass, and I tried to break it (empty input, huge input, fake drug,
   prompt injection, two users at once).
7. Change one thing, measure, then change the next. Keep the before/after numbers.
