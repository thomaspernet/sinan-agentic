---
mandatory_for:
  universal: true
  rules: [critical]
---

# Clean Code

The principles that guide every decision. Language-agnostic, framework-agnostic. Read once, apply everywhere.

---

## 1. No Backward Compatibility

When you improve something, improve it completely. Do not leave the old version around "just in case." Do not add shims, adapters, or re-exports to keep old call sites working. Do not rename unused parameters to `_old_name` so the function signature doesn't change.

If a function is renamed, update every caller. If a data model changes, migrate the data. If an interface changes, update every implementation. The cost of updating is linear and bounded. The cost of maintaining two parallel paths grows forever.

Backward compatibility is appropriate for public APIs with external consumers you cannot control. Inside a codebase you own, it is technical debt disguised as caution.

## 2. No Shortcuts

Every shortcut is a loan against the future. The interest is confusion, inconsistency, and cascading fixes.

Fix root causes, not symptoms. If the data is wrong, fix it where it is created, not where it is displayed. If a function is too complex, redesign it — do not add a wrapper. If a test is flaky, find the race condition — do not add a retry.

No TODO comments that live forever. No "temporary" workarounds. No copy-pasted blocks with minor variations. No magic numbers with an explanatory comment instead of a named constant. No disabling a linter rule because the code doesn't pass.

If a fix would take too long right now, create an issue. Do not leave a patch in the code.

## 3. No Duplication

Every piece of knowledge should have a single, authoritative representation. When the same logic exists in two places, one of them will eventually be wrong.

This applies to code, data, configuration, and documentation. If two functions do the same thing, extract the shared logic. If two config files define the same default, consolidate into one. If a constant appears as a string literal in multiple queries, define it once and reference it everywhere.

Duplication is not just identical lines. It is also parallel structures that encode the same decision. Two switch statements branching on the same condition, two API modules building the same URL pattern, two components formatting dates differently — these are all duplication. The knowledge of how to do that thing should live in one place.

Before writing a new utility, check if one already exists. Scattered one-off helpers across random files is a form of duplication — the same kind of logic in multiple places, each slightly different.

## 4. Refactor Forward

When you encounter messy code, make it better. Do not add a clean layer on top of a dirty layer — that just creates two layers to maintain.

If the existing code does not follow current conventions, update it. If a function has grown too long, split it. If a module has too many responsibilities, separate them. If naming is inconsistent, rename.

This does not mean rewrite everything you touch. It means leave code better than you found it. Small, incremental improvements compound. Patches on patches compound too — in the wrong direction.

## 5. Forward-Looking, Not Speculative

Design for what you know, not for what you imagine. Write the code that solves today's problem cleanly. Do not add extension points, plugin systems, or configurability for requirements that do not exist yet.

Three similar lines of code is better than a premature abstraction. A concrete implementation is better than a generic framework with one user. The right amount of complexity is what the task actually requires — no speculative abstractions, but no half-finished implementations either.

This is not a license to write rigid code. Good design naturally accommodates change — clear interfaces, small functions, and separated concerns are inherently extensible. The difference is between code that is easy to change (good) and code that anticipates specific changes that may never come (speculative). Make it easy to add configuration later, but do not add the configuration until someone needs it.

## 6. Decouple Everything

Every component should be independent and replaceable without rippling through the rest of the system. This applies at every boundary — layers, modules, vendors, data formats, workflows, and configuration.

Each layer communicates through a defined interface. Services do not know how the database stores data. The API layer does not know how services implement logic. Infrastructure does not know what business rules exist. Change one layer and the others do not notice. Within layers, modules do not reach into each other's internals — if service A needs something from service B, it calls B's public interface, not B's private state.

External dependencies deserve the same treatment. Wrap vendor services behind interfaces you own. Define your own data types at the boundary — do not let a vendor's response schema leak into your domain. Keep vendor-specific code in dedicated adapter modules. This does not mean abstracting everything — your language, framework, and database are foundational choices. Decouple from things you might realistically replace: API providers, file parsers, embedding models, authentication services.

Data format boundaries get exactly one transformation point. The backend owns its naming convention. The frontend owns its naming convention. The conversion happens in one place, not scattered across every component. Database schemas do not leak into API responses. Vendor response shapes do not leak into domain models. Each system speaks its own language internally and translates at the edge.

Workflow steps have defined contracts — what a step receives and what it produces. A step does not reach into the previous step's internals or assume anything about the next step's implementation. If step B fails, step A's output is still valid. If step C changes its logic, step B does not need to change its output. Steps are independently testable, retryable, and replaceable.

How something is organized is separate from what it is. Moving an item in a hierarchy does not modify the item. Changing a grouping strategy does not alter the underlying data. Renaming a category does not touch the records in it. Structure is metadata about content, not a property of it.

Business rules that might change — processing options, model selections, feature flags, thresholds, timeouts, page sizes — live in configuration, not in code. Code defines the mechanism. Configuration defines the policy. If a product decision changes, you change a config value, not a line of code.

The test is always the same: if I change X, how many other things break? If the answer is "just X," you are decoupled. If the answer is "X and everything that touches X," you have a hidden dependency to remove. Decoupling is not about adding abstraction layers. It is about drawing clear boundaries. A well-decoupled system has fewer abstractions, not more — because each component does one thing and the boundaries between them are obvious.

## 7. Organize by Family

Code is grouped by what it belongs to, not dumped into flat catch-all directories. When you add a file, it goes into the folder for its feature, domain, or family — not into a growing pile of unrelated siblings.

A flat `utils/` with forty unrelated helpers, a `models/` directory holding every entity in the system, a `components/` folder with sixty files at one level — these hide structure. The layout of a codebase should reveal its domains at a glance. Someone opening the tree should see the features, not a wall of files.

Group first, then nest as a family grows. Start by putting siblings that change together under one roof — everything chat-related in one place, every auth entity together. When a group gets large enough that you scan past unrelated files to find what you want, split it into sub-packages. Depth is a tool, not a goal: add a level when it earns its keep, not preemptively.

The test: when you create a file, is there an obvious folder it belongs in? If yes, put it there. If the only home is a flat catch-all bucket, the structure is missing a grouping — add it. This is the spatial side of decoupling: a tree grouped by domain makes duplication, dead code, and misplaced responsibilities visible, where a flat dump hides them. (Language-specific layouts: see the backend and frontend architecture docs.)

---

## How These Principles Work Together

These seven principles reinforce each other. No backward compatibility forces you to refactor forward — you cannot improve something halfway and leave the old version around. Refactoring forward eliminates duplication — you fix the root issue instead of patching around it. Eliminating duplication removes the temptation for shortcuts — when logic lives in one place, there is nothing to copy-paste. Avoiding shortcuts keeps the code honest — every decision is deliberate, not expedient. Decoupling ensures that each principle can be applied locally — fixing one module does not cascade into unrelated modules. Forward-looking design ensures you do not add complexity in the name of preventing it — you solve today's problem and trust that good structure makes tomorrow's changes easy. Organizing by family makes the other six visible — duplication, hidden dependencies, and dead code stand out in a tree grouped by domain and disappear into a flat dump.

When two principles seem to conflict, the resolution is usually scope. "No duplication" does not mean building a generic framework for two similar functions — that would be speculative. "Refactor forward" does not mean rewriting an entire module to fix a one-line bug — that would be a shortcut away from the actual task. "Decouple everything" does not mean wrapping every function call in an interface — that would be speculative abstraction. Apply each principle within the scope of the work at hand.
