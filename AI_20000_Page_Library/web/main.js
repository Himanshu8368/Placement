import "./style.css";

import chaptersData from "./data/chapters.json";

/** @type {{ collections: Array<{id: string, name: string, description: string, parts: Array<{id: string, title: string, sections: Array<{id: string, title: string, chapters: Array<{id: string, title: string, status: string, contentHtml: string}>}>}>}> }} */
const data = chaptersData;

function getCollections() {
  return data.collections;
}

function findCollection(collectionId) {
  return getCollections().find((c) => c.id === collectionId);
}

function findChapter(collectionId, chapterId) {
  const collection = findCollection(collectionId);
  if (!collection) return null;
  for (const part of collection.parts ?? []) {
    for (const section of part.sections ?? []) {
      const chapter = (section.chapters ?? []).find((c) => c.id === chapterId);
      if (chapter) {
        return {
          ...chapter,
          partTitle: part.title,
          sectionTitle: section.title,
        };
      }
    }
  }
  return null;
}

function getRoute() {
  const hash = window.location.hash.replace(/^#\/?/, "");
  const [page, a, b] = hash.split("/");
  return { page, a, b };
}

function appShell(content) {
  return `
    <header class="topbar">
      <a href="#/" class="brand">Knowledge Library</a>
      <nav class="topnav">
        <a href="#/collection/ai">AI</a>
        <a href="#/collection/dsa">DSA</a>
        <a href="#/collection/system-design">System Design</a>
      </nav>
    </header>
    <main class="container">${content}</main>
  `;
}

function homePage() {
  return appShell(`
    <section class="hero">
      <h1>Rewritten Knowledge Hub</h1>
      <p>Migrate chapter-by-chapter from Markdown to modern web pages while keeping your original content safe.</p>
    </section>
    <section class="cards">
      ${getCollections()
        .map(
          (item) => `
        <a class="card" href="#/collection/${item.id}">
          <h2>${item.name}</h2>
          <p>${item.description}</p>
          <span class="cta">Open collection</span>
        </a>
      `
        )
        .join("")}
    </section>
  `);
}

function collectionPage(collectionId) {
  const collection = findCollection(collectionId);
  if (!collection) {
    return notFoundPage();
  }

  return appShell(`
    <section class="page-head">
      <h1>${collection.name}</h1>
      <p>${collection.description}</p>
      <a class="back" href="#/">Back to home</a>
    </section>
    <section class="chapter-list">
      ${(collection.parts ?? [])
        .map(
          (part) => `
        <div class="tree-group">
          <h2 class="tree-title">${part.title}</h2>
          ${(part.sections ?? [])
            .map(
              (section) => `
            <div class="tree-section">
              <h3 class="tree-subtitle">${section.title}</h3>
              <div class="tree-chapters">
                ${(section.chapters ?? [])
                  .map(
                    (ch) => `
                  <a class="chapter-card" href="#/chapter/${collectionId}/${ch.id}">
                    <div>
                      <h3>${ch.title}</h3>
                      <p>${part.title} • ${section.title}</p>
                    </div>
                    <span class="status ${ch.status}">${ch.status}</span>
                  </a>
                `
                  )
                  .join("")}
              </div>
            </div>
          `
            )
            .join("")}
        </div>
      `
        )
        .join("")}
    </section>
  `);
}

function chapterPage(collectionId, chapterId) {
  const chapter = findChapter(collectionId, chapterId);

  if (!chapter) {
    return notFoundPage();
  }

  return appShell(`
    <article class="chapter">
      <a class="back" href="#/collection/${collectionId}">Back to collection</a>
      <h1>${chapter.title}</h1>
      <p class="meta">${chapter.partTitle} • ${chapter.sectionTitle}</p>
      <div class="chapter-content">${chapter.contentHtml}</div>
    </article>
  `);
}

function notFoundPage() {
  return appShell(`
    <section class="notfound">
      <h1>Page not found</h1>
      <p>The route you opened does not exist yet.</p>
      <a class="back" href="#/">Go to home</a>
    </section>
  `);
}

function render() {
  const { page, a, b } = getRoute();
  const app = document.querySelector("#app");

  if (!page) {
    app.innerHTML = homePage();
    return;
  }

  if (page === "collection") {
    app.innerHTML = collectionPage(a);
    return;
  }

  if (page === "chapter") {
    app.innerHTML = chapterPage(a, b);
    return;
  }

  app.innerHTML = notFoundPage();
}

window.addEventListener("hashchange", render);
render();
