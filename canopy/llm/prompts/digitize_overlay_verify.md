
You are checking someone else's figure read-out. The image is the original figure with numbered
marks drawn on top of it. Each mark claims to sit on a particular datum.

## The target that was read

{{TARGET}}

## What each mark claims

{{MARKS}}

## Your job

For every numbered mark, say whether the mark is where its claim says it should be:

* `ok` — the mark sits on the datum it claims (within a marker's width).
* `not_on_datum` — the right series and x position, but the mark misses the plotted value
  (it sits above/below the marker centre or the bar top, on the error bar, or on nothing).
* `wrong_series` — the mark is on a different group's series than it claims.
* `wrong_x` — the mark is at a different x position / block / time point than it claims.
* `unknown` — the mark is unreadable at this size, or the figure genuinely does not settle it.

Report one entry per mark, using the number printed beside it. Be specific in `reason`: name what
the mark is actually sitting on. Do not re-read the values and do not propose corrections — only
judge placement. Zoom with `crop_image` before judging a crowded mark.

If several marks land on the same pixel they are drawn on top of each other; judge that position
once and give the same verdict to each number that claims it.

Finish by calling the `submit` tool. Never answer in prose.
