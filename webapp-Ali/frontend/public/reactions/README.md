# Reaction clips

Display-only reaction clips, played after the **factuality** prediction renders.
These are an OUTPUT keyed to the predicted label — never an input to any model.

Drop one file per factuality label here (served from `/reactions/<file>`):

| Factuality label | File              |
| ---------------- | ----------------- |
| VERY LOW         | `very_low.mp4`    |
| LOW              | `low.mp4`         |
| MIXED            | `mixed.mp4`       |
| HIGH             | `high.mp4`        |
| VERY HIGH        | `very_high.mp4`   |

The mapping lives in `frontend/src/types.ts` (`REACTION_CLIPS`). A missing file
degrades gracefully — the clip area simply doesn't render (no crash).
