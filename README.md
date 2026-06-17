# Personal Blog

A personal blog built with [Astro](https://astro.build), based on the official blog starter template with enhanced features for better reading experience.

## ✨ Enhanced Features

- ✅ **Table of Contents Navigation** - Collapsible sidebar with h1/h2/h3 headings, auto-highlight current section
- ✅ **LaTeX Math Rendering** - Full KaTeX support for inline and block equations via remark-math
- ✅ **Refined Typography** - Optimized spacing, table styling, and prose layout for readability
- ✅ **Local Asset Management** - Blog images stored locally for reliability
- ✅ Minimal styling (customized from Bear Blog theme)
- ✅ 100/100 Lighthouse performance
- ✅ SEO-friendly with canonical URLs and Open Graph data
- ✅ Sitemap and RSS Feed support
- ✅ Markdown & MDX support

## 🚀 Project Structure

```text
├── public/
├── src/
│   ├── assets/
│   ├── components/
│   │   ├── TableOfContents.astro  # TOC navigation component
│   │   └── ...
│   ├── content/
│   │   └── blog/                  # Blog posts with local images
│   ├── layouts/
│   │   └── BlogPost.astro         # Blog post layout with TOC integration
│   ├── pages/
│   └── styles/
│       └── global.css             # Enhanced typography and table styles
├── astro.config.mjs               # KaTeX plugins configured here
└── package.json
```

## 🧞 Commands

All commands are run from the root of the project:

| Command                   | Action                                           |
| :------------------------ | :----------------------------------------------- |
| `npm install`             | Installs dependencies                            |
| `npm run dev`             | Starts local dev server at `localhost:4321`      |
| `npm run build`           | Build your production site to `./dist/`          |
| `npm run preview`         | Preview your build locally, before deploying     |
| `npm run astro ...`       | Run CLI commands like `astro add`, `astro check` |
| `npm run astro -- --help` | Get help using the Astro CLI                     |

## Credit

This theme is based on the [Astro Blog Starter](https://github.com/withastro/astro/tree/main/examples/blog) and [Bear Blog](https://github.com/HermanMartinus/bearblog/).
