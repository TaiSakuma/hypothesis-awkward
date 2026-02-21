# Git

- Do not stage or commit automatically. Let the user review changes first.
- When the user asks to stage and commit, run `git add` and `git commit` as
  separate steps so the user can review the staged changes before committing.
- When the user asks to commit, show the proposed commit message and ask for
  confirmation before running `git commit`.
