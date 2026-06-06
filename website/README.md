# SINCPS project site

A small project page for SINCPS (Semantic-aware Implicit Neural Compression for
Physics Simulations), styled to match the personal portfolio
(Vite + React + Tailwind v4 + framer-motion). All page content lives in
`src/data/project.js`, so it can be extended by editing data, not JSX.

Target URL: https://jezemba.github.io/SINCPS

## Local development (needs Node 18+)
```
npm install
npm run dev      # local preview
npm run build    # outputs to dist/
```

## Deploy (no local Node required)
This repo includes a GitHub Actions workflow (`.github/workflows/deploy.yml`) that
builds the site and publishes it to GitHub Pages on every push to `main`.

One-time setup in the GitHub repo: Settings > Pages > Build and deployment >
Source = "GitHub Actions". Then push to `main` and the site goes live at the URL
above.

## Notes
- `vite.config.js` sets `base: '/SINCPS/'` for a project page. If you switch to a
  separate repo root or a custom domain, change `base` and
  `public/404.html` (pathSegmentsToKeep).
- Add more figures by dropping images in `src/assets` and appending to the
  `images` array in `src/data/project.js`.
