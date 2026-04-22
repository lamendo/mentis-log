# Release Handoff — v0.1.0

Everything is prepared locally in `C:\mentis_log\`. This file lists
the exact commands **you** need to run to publish the release.

Nothing below has been executed automatically.

## State of the repo

- **Location**: `C:\mentis_log\` (fresh git repo, not connected to
  `mentis_ai`)
- **Branch**: `main`
- **Commits**: 1 (`Initial public release (v0.1.0)`)
- **Files**: 66 tracked, ~2.0 MB
- **Tests**: 207 passing locally
- **Entry point**: `mentis-log` (installed via `pip install -e .`)

## 1. Create the GitHub repo

You own `github.com/lamendo`. Create an **empty** repo there named
`mentis-log` (no README, no LICENSE, no .gitignore — all of those
are already in the local repo).

Option A — via `gh` CLI (if installed):

```bash
cd C:\mentis_log
gh repo create lamendo/mentis-log --public --source=. --remote=origin --push
```

Option B — manually on github.com:

1. Go to https://github.com/new
2. Owner: `lamendo`
3. Repository name: `mentis-log`
4. Public
5. Do NOT tick "Add a README / .gitignore / license"
6. Create
7. Then:
   ```bash
   cd C:\mentis_log
   git remote add origin https://github.com/lamendo/mentis-log.git
   git push -u origin main
   ```

## 2. Verify CI runs green

After the first push, GitHub Actions will execute
`.github/workflows/ci.yml` across Python 3.10 / 3.11 / 3.12 / 3.13.

Wait ~5 minutes, then check:

```
https://github.com/lamendo/mentis-log/actions
```

All four matrix cells must pass. If one fails, fix it locally and
push the correction before tagging.

## 3. Tag v0.1.0 and create the GitHub Release

```bash
cd C:\mentis_log
git tag -a v0.1.0 -m "v0.1.0 — initial public release"
git push --tags
```

Then create the GitHub Release tied to that tag:

Option A — `gh` CLI:

```bash
gh release create v0.1.0 \
    --title "v0.1.0 — initial public release" \
    --notes-file CHANGELOG.md
```

Option B — https://github.com/lamendo/mentis-log/releases/new:

1. Choose tag: `v0.1.0`
2. Release title: `v0.1.0 — initial public release`
3. Description: paste the content of `CHANGELOG.md` (just the
   `## [0.1.0]` section)
4. Publish

GitHub automatically attaches source tarball + zip as release assets.

## 4. PyPI upload (first time)

### One-time setup

- Register a PyPI account: https://pypi.org/account/register/
- Enable 2FA (required for first upload).
- Create a **project-scoped API token** once you have uploaded once
  manually — but for the very first upload you use a **global token**:
  - https://pypi.org/manage/account/token/ → Create
  - Scope: "Entire account" (for first upload only)
  - Copy the token (starts with `pypi-`)

Store it in `%USERPROFILE%\.pypirc`:

```ini
[pypi]
username = __token__
password = pypi-AgENdGVzdC5weXBp...your token here...
```

### Build and upload

```bash
cd C:\mentis_log
.venv\Scripts\pip install --upgrade build twine
.venv\Scripts\python -m build
.venv\Scripts\twine check dist\*
.venv\Scripts\twine upload dist\*
```

`python -m build` produces both a source tarball (`.tar.gz`) and a
wheel (`.whl`) under `dist/`. `twine check` validates metadata.
`twine upload` pushes to PyPI.

After success:

```
https://pypi.org/project/mentis-log/
```

And `pip install mentis-log` now works worldwide.

### Test-PyPI first (optional but recommended)

If you want to dry-run before hitting real PyPI:

```bash
# Test-PyPI token from https://test.pypi.org/manage/account/token/
.venv\Scripts\twine upload --repository testpypi dist\*

# Verify install:
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ mentis-log
```

If that works, repeat without `--repository testpypi` against
production PyPI.

## 5. Post-release polish (optional)

- **Add a GitHub profile README**: create a personal repo named
  `lamendo/lamendo`, add a `README.md` there; it will be shown on
  your GitHub profile page. Reduces the "anonymous account" feel for
  visitors to `github.com/lamendo/mentis-log`.
- **Pin the release**: on the repo landing page, use "Customize your
  pins" to feature `mentis-log`.
- **Add a topic**: on the repo page → About → Topics →
  `log-analysis`, `segmentation`, `change-point-detection`,
  `observability`.
- **Author email in `pyproject.toml`** is currently
  `jan.reinhard55@gmail.com`. PyPI shows this publicly. If you want
  to shield it, change the `authors` line in `pyproject.toml` to use
  a `noreply` address **before** the first PyPI upload, e.g.:
  ```toml
  authors = [{ name = "Jan Reinhard" }]
  ```
  (email omitted entirely is allowed). Then rebuild and upload.

## 6. Things I deliberately did NOT do

- Push to any remote
- Create any GitHub / PyPI account or token
- Advertise the repo anywhere

Those are all your calls.

## Rollback if something goes wrong

Before first tag / PyPI upload: just delete `C:\mentis_log\` and
restart from the extract folder — nothing was pushed.

After GitHub push but before PyPI: delete the GitHub repo (Settings
→ Danger Zone → Delete). The URL stays reserved under your account
for a while but can be reused.

After PyPI upload: the version `0.1.0` is **permanently reserved**
on PyPI — you cannot reuse it. You can `pip yank` it to hide it
from `pip install`, but never re-upload 0.1.0. The fix is to
release `0.1.1`.

## Sanity checklist before each step

Before push:

- [ ] `pytest tests/ -q` — all 207 pass
- [ ] `git status` clean
- [ ] `git log --oneline` shows exactly one commit

Before tag:

- [ ] CI green on all four Python versions
- [ ] CHANGELOG reflects what shipped

Before PyPI upload:

- [ ] `python -m build` succeeded
- [ ] `twine check dist/*` → `PASSED` on every file
- [ ] You've decided on the author email (public on PyPI)
- [ ] Token is scoped correctly
