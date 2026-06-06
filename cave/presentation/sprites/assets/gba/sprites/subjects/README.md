# Subject sprite variants

The renderer looks for frames in `sprites/subjects/<subject-id>/` first and
falls back to `sprites/subjects/default/` for any missing frame.

Expected game subject ids:

- `sleepy`
- `excited`
- `adhd`
- `artsy`

Keep the same filenames across variants, for example `subject_walk_0.png`,
`subject_expect_0.png`, and `subject_surprised_0.png`.
