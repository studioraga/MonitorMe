# Assistant fact guard

MonitorMe uses a conservative fact guard before returning or saving assistant answers.

## Allowed v0.1 facts

Allowed facts are facts present in normalized local evidence:

- event type
- object label
- confidence
- bbox
- frame id
- session id
- camera id
- model id
- artifact path
- policy decision
- audit id

## Blocked v0.1 facts

These are blocked unless future normalized evidence explicitly supports them:

- face recognition
- identity/name
- weapon claims
- criminal/suspicious intent
- gender/age inference
- detailed clothing description without VLM evidence

## Example

Question:

```text
Was the person carrying a weapon?
```

If the DB only contains `label=person`, MonitorMe must answer:

```text
I do not have local evidence for that request.
```

It must not invent a weapon, identity, motive, or visual detail.
