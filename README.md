# Doctor Trigger — Project Page

Static site (plain HTML/CSS/JS, no build step). The interactive 3D panel in the Results
section loads [three.js](https://threejs.org/) from cdnjs at runtime; that is the only
external dependency.

```
index.html                # the whole page
static/images/*.png       # figures cropped from the paper
static/videos/*.mp4       # 3-minute overview video, plus its poster frame (video_poster.jpg)
static/pdf/paper.pdf      # paper PDF linked from the "Paper" button
.nojekyll                 # tell GitHub Pages to serve files as-is
```

Preview locally:

```bash
python3 -m http.server 8000   # then open http://localhost:8000
```

## Deploy

### During double-blind review (anonymous)
1. Push this folder to a GitHub repo (can be private).
2. Go to https://anonymous.4open.science → "Anonymize your repository", pick the repo,
   add terms to redact (your name, lab, GitHub username), and enable **"GitHub Pages / website"**.
3. You get a link like `https://anonymous.4open.science/w/<your-id>/` to put in the paper.

> Do not use `username.github.io` before the review period ends: the URL reveals your identity.

### After acceptance (GitHub Pages)
1. Create a repo, e.g. `doctor-trigger`, and push these files to the root of `main`.
2. Repo → Settings → Pages → Source: *Deploy from a branch* → `main` / `/ (root)` → Save.
3. The site appears at `https://<username>.github.io/doctor-trigger/` after about a minute.

## To update after de-anonymization
Search `TODO` in `index.html`: authors/affiliations, OpenReview/arXiv link, code link, BibTeX.
