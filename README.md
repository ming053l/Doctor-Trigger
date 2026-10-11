# Doctor Trigger — Project Page

Static site (plain HTML/CSS/JS, no build step). The interactive 3D panel in the Results
section loads [three.js](https://threejs.org/) from cdnjs at runtime; that is the only
external dependency.

```
index.html                       # the whole page
static/images/fig*.png           # figures cropped from the paper
static/images/teaser_poster.jpg  # poster frame of the hero teaser
static/images/promo_poster.jpg   # poster frame of the 83-second short film (shared by EN and ZH)
static/images/video_poster.jpg   # poster frame of the 3-minute overview
static/images/compare_*.png      # drag-to-compare pair: original / posted (AI-generated face, 512 px)
static/videos/teaser.*           # 22-second silent looping teaser in the hero band (mp4 + webm)
static/videos/promo_en.mp4       # 83-second short film, English (main video)
static/videos/promo_zh.mp4       # 83-second short film, Chinese (中文版 tab)
static/videos/doctor_trigger.mp4 # 3-minute overview video
static/pdf/paper.pdf             # paper PDF linked from the "Paper" button
.nojekyll                        # tell GitHub Pages to serve files as-is
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
Search `TODO` in `index.html`: the OpenReview/arXiv link (the disabled pill in the header). Update the BibTeX `note` once there is a venue.
