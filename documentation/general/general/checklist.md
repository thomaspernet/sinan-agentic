---
mandatory_for:
  universal: true
  rules: [critical]
---

# Completion Checklist

Run through this checklist before marking any task as done. Every item is a yes/no question. If you are unsure about a check, follow the link to the full rule.

This is the general checklist — language-agnostic. After this, run the language-specific checklist ([Python](python/checklist) or [TypeScript](typescript/checklist)).

---

## Scope

- [ ] Every change traces back to the issue. No extra features, no drive-by refactors, no "while I'm here" additions. → [No Shortcuts](clean-code#2-no-shortcuts)
- [ ] No unrelated files modified. If you found something else to fix, create a separate issue. → [No Shortcuts](clean-code#2-no-shortcuts)
- [ ] Propagation decision made. If the diff adds a helper, pattern, perf fix, or bug-fix shape, run `/propagation-scan` to file sibling sites as their own issues. If the diff is purely local, cosmetic, or a dependency bump, skip the scan. → [Propagation scan](propagation-scan)

## Code Quality

- [ ] No hardcoded values. Thresholds, limits, timeouts, page sizes, model names, feature flags — all in configuration, not in code. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] No magic strings. Entity names, table names, node labels, relationship types, status values — all behind constants or enums. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] No duplication. Checked for existing helpers, utilities, and shared functions before writing new ones. → [Pillar 3: No Duplication](clean-code#3-no-duplication)
- [ ] No backward compatibility hacks. Old code removed, not shimmed. No re-exports, no `_old_name` renames, no `// removed` comments. → [Pillar 1: No Backward Compatibility](clean-code#1-no-backward-compatibility)
- [ ] No shortcuts or patches. Root cause fixed, not symptoms. No TODO comments left behind — create an issue instead. → [Pillar 2: No Shortcuts](clean-code#2-no-shortcuts)
- [ ] No speculative abstractions. No extension points, plugin systems, or configurability for requirements that do not exist yet. → [Pillar 5: Forward-Looking, Not Speculative](clean-code#5-forward-looking-not-speculative)

## Structure

- [ ] Files grouped by family/feature into the right subfolder, not dumped flat into a shared catch-all dir. New files live with the thing they belong to. → [Pillar 7: Organize by Family](clean-code#7-organize-by-family)

## Decoupling

- [ ] Layer boundaries respected. No business logic in the API layer. No database queries outside the repository layer. No HTTP concerns in services. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] Modules do not reach into each other's internals. Cross-module calls go through public interfaces. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] Vendor code isolated in adapter modules. No vendor response shapes leaking into domain models. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] Data format transforms happen at exactly one boundary, not scattered across multiple files. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] Workflow steps have clear contracts. Each step receives and produces defined types. No step reaches into another step's internals. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)
- [ ] Configuration separated from code. Policy decisions changeable without code changes. → [Pillar 6: Decouple Everything](clean-code#6-decouple-everything)

## Cleanup

- [ ] No unused code left behind. Dead imports, commented-out blocks, unreachable branches — deleted, not commented. → [Pillar 4: Refactor Forward](clean-code#4-refactor-forward)
- [ ] Code left better than found. If you touched messy code, you improved it. If naming was inconsistent, you fixed it. → [Pillar 4: Refactor Forward](clean-code#4-refactor-forward)

## Tests

- [ ] Tests written for new behavior.
- [ ] All tests passing.
- [ ] No test workarounds (retries, sleeps, skipped assertions).
