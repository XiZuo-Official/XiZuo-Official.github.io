import { LANGUAGE_OPTIONS, DEFAULT_LOCALE } from "/config/languages.js";
import { TEXTS } from "/config/i18n.js";
import { SITE_DATA } from "/config/site-data.js";
import { NOTES } from "/data/notes.js";
import { BLOG_POSTS } from "/data/blog-posts.js";
import { markdownToHtml, stripMarkdown } from "/scripts/markdown.js";

const state = {
  locale: DEFAULT_LOCALE,
  notes: [],
  blogs: [],
  activeNoteId: null,
  globalQuery: "",
  noteQuery: ""
};

const enabledLanguages = LANGUAGE_OPTIONS.filter((item) => item.enabled);

const els = {
  languageSwitcher: document.querySelector("#language-switcher"),
  nav: document.querySelector("#main-nav"),
  pages: [...document.querySelectorAll(".page")],
  introText: document.querySelector("#intro-text"),
  avatar: document.querySelector("#avatar"),
  emailLink: document.querySelector("#email-link"),
  onlyfansLink: document.querySelector("#onlyfans-link"),
  alipayQr: document.querySelector("#alipay-qr"),
  wechatQr: document.querySelector("#wechat-qr"),
  zelleQr: document.querySelector("#zelle-qr"),
  globalSearchInput: document.querySelector("#global-search"),
  globalSearchButton: document.querySelector("#global-search-btn"),
  globalSearchResult: document.querySelector("#global-search-result"),
  notesList: document.querySelector("#notes-list"),
  noteTitle: document.querySelector("#note-title"),
  noteContent: document.querySelector("#note-content"),
  noteSearchWrap: document.querySelector("#note-search-wrap"),
  noteSearchInput: document.querySelector("#note-search"),
  noteSearchButton: document.querySelector("#note-search-btn"),
  noteSearchResult: document.querySelector("#note-search-result"),
  noteStructure: document.querySelector("#note-structure"),
  noteToc: document.querySelector("#note-toc"),
  blogList: document.querySelector("#blog-list")
};

function getText(key, vars = {}) {
  const current = TEXTS[state.locale] || TEXTS[DEFAULT_LOCALE];
  const fallback = TEXTS[DEFAULT_LOCALE];
  let value = current[key] || fallback[key] || key;

  for (const [k, v] of Object.entries(vars)) {
    value = value.replace(`{${k}}`, String(v));
  }

  return value;
}

function getLocalizedValue(value) {
  if (value && typeof value === "object") {
    return value[state.locale] || value[DEFAULT_LOCALE] || Object.values(value)[0] || "";
  }
  return value || "";
}

function resolveLocalizedFile(filesLike) {
  if (!filesLike) return "";
  if (typeof filesLike === "string") return filesLike;

  const candidates = [state.locale, DEFAULT_LOCALE, "ja", "en", "zh-CN", ...Object.keys(filesLike)];
  for (const code of candidates) {
    if (filesLike[code]) return filesLike[code];
  }

  return "";
}

function createSakuraPetal() {
  const layer = document.querySelector(".sakura-layer");
  if (!layer) return;

  const petal = document.createElement("span");
  petal.className = "petal";

  const startLeft = Math.random() * 100;
  const drift = `${(Math.random() * 40 - 20).toFixed(1)}vw`;
  const duration = 7 + Math.random() * 9;

  petal.style.left = `${startLeft}vw`;
  petal.style.setProperty("--drift", drift);
  petal.style.animationDuration = `${duration.toFixed(1)}s`;

  layer.appendChild(petal);
  setTimeout(() => petal.remove(), duration * 1000);
}

function startSakura() {
  for (let i = 0; i < 18; i += 1) {
    setTimeout(createSakuraPetal, i * 220);
  }
  setInterval(createSakuraPetal, 420);
}

function renderLanguageSwitcher() {
  els.languageSwitcher.innerHTML = "";

  enabledLanguages.forEach((lang) => {
    const button = document.createElement("button");
    button.className = `lang-btn tap-bounce${lang.code === state.locale ? " active" : ""}`;
    button.textContent = lang.label;
    button.type = "button";

    button.addEventListener("click", async () => {
      if (lang.code === state.locale) return;
      state.locale = lang.code;
      localStorage.setItem("preferred-locale", lang.code);
      await reloadLocalizedContent();
      refreshAllViews();
    });

    els.languageSwitcher.appendChild(button);
  });
}

function applyI18n() {
  document.documentElement.lang = state.locale;

  document.querySelectorAll("[data-i18n]").forEach((node) => {
    const key = node.getAttribute("data-i18n");
    node.textContent = getText(key);
  });

  els.globalSearchInput.placeholder = getText("searchAllPlaceholder");
  els.noteSearchInput.placeholder = getText("searchInNotePlaceholder");
  els.introText.textContent = getLocalizedValue(SITE_DATA.profile.intro);

  if (!state.activeNoteId) {
    els.noteTitle.textContent = getText("noteEmpty");
  }
}

function setProfile() {
  els.avatar.src = SITE_DATA.profile.avatar;
  els.avatar.onerror = () => {
    els.avatar.src = "/assets/avatar-placeholder.svg";
  };

  els.alipayQr.src = SITE_DATA.donate.alipayQr;
  els.alipayQr.onerror = () => {
    els.alipayQr.src = "/assets/alipay-placeholder.svg";
  };
  els.wechatQr.src = SITE_DATA.donate.wechatQr;
  els.wechatQr.onerror = () => {
    els.wechatQr.src = "/assets/wechat-placeholder.svg";
  };
  els.zelleQr.src = SITE_DATA.donate.zelleQr;
  els.zelleQr.onerror = () => {
    els.zelleQr.src = "/assets/zelle-placeholder.svg";
  };

  els.emailLink.href = SITE_DATA.profile.email;
  els.onlyfansLink.href = SITE_DATA.profile.onlyfans.href;
}

async function fetchTextContent(filePath) {
  if (!filePath) return "";
  const response = await fetch(filePath);
  if (!response.ok) throw new Error(`Failed to load file: ${filePath}`);
  return response.text();
}

async function loadNotes() {
  const tasks = NOTES.map(async (note) => {
    const filePath = resolveLocalizedFile(note.files || note.file);
    let raw = "";

    try {
      raw = await fetchTextContent(filePath);
    } catch (error) {
      console.error(error);
    }

    return {
      ...note,
      filePath,
      raw,
      plain: stripMarkdown(raw)
    };
  });

  state.notes = await Promise.all(tasks);

  if (!state.activeNoteId || !state.notes.some((item) => item.id === state.activeNoteId)) {
    state.activeNoteId = state.notes[0]?.id || null;
  }
}

async function loadBlogs() {
  const tasks = BLOG_POSTS.map(async (post) => {
    let content = "";

    if (post.files) {
      const filePath = resolveLocalizedFile(post.files);
      try {
        content = await fetchTextContent(filePath);
      } catch (error) {
        console.error(error);
      }
    } else {
      content = getLocalizedValue(post.content);
    }

    return {
      ...post,
      localizedContent: content
    };
  });

  state.blogs = await Promise.all(tasks);
}

async function reloadLocalizedContent() {
  await Promise.all([loadNotes(), loadBlogs()]);
}

function getFilteredNotes() {
  const query = state.globalQuery.trim().toLowerCase();
  if (!query) return state.notes;
  return state.notes.filter((note) => note.plain.toLowerCase().includes(query));
}

function renderNotesList() {
  const notes = getFilteredNotes();
  els.notesList.innerHTML = "";

  notes.forEach((note) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `note-item tap-bounce${note.id === state.activeNoteId ? " active" : ""}`;
    button.textContent = getLocalizedValue(note.title);

    button.addEventListener("click", async () => {
      state.activeNoteId = note.id;
      state.noteQuery = "";
      els.noteSearchInput.value = "";
      renderNotesList();
      await renderActiveNote();
    });

    const li = document.createElement("li");
    li.appendChild(button);
    els.notesList.appendChild(li);
  });

  if (!notes.length) {
    els.globalSearchResult.textContent = getText("searchNone");
  } else if (state.globalQuery.trim()) {
    els.globalSearchResult.textContent = getText("searchAllResult", { count: notes.length });
  } else {
    els.globalSearchResult.textContent = "";
  }
}

function clearMarks(container) {
  const marks = container.querySelectorAll("mark");
  marks.forEach((mark) => {
    const parent = mark.parentNode;
    if (!parent) return;
    parent.replaceChild(document.createTextNode(mark.textContent || ""), mark);
    parent.normalize();
  });
}

function highlightInElement(container, query) {
  clearMarks(container);
  const keyword = query.trim();
  if (!keyword) return 0;

  const regex = new RegExp(keyword.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"), "gi");
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT);
  const nodes = [];

  while (walker.nextNode()) {
    const textNode = walker.currentNode;
    if (textNode.nodeValue && regex.test(textNode.nodeValue)) {
      nodes.push(textNode);
    }
    regex.lastIndex = 0;
  }

  let count = 0;

  nodes.forEach((node) => {
    if (node.parentElement?.closest(".mjx-container")) return;

    const text = node.nodeValue || "";
    if (!text.trim()) return;

    const frag = document.createDocumentFragment();
    let last = 0;

    text.replace(regex, (match, offset) => {
      if (offset > last) {
        frag.appendChild(document.createTextNode(text.slice(last, offset)));
      }
      const mark = document.createElement("mark");
      mark.textContent = match;
      frag.appendChild(mark);
      last = offset + match.length;
      count += 1;
      return match;
    });

    if (last < text.length) {
      frag.appendChild(document.createTextNode(text.slice(last)));
    }

    if (node.parentNode) node.parentNode.replaceChild(frag, node);
  });

  return count;
}

function slugifyHeading(text, index, used) {
  const base = text
    .toLowerCase()
    .trim()
    .replace(/[^\w\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\s-]/g, "")
    .replace(/\s+/g, "-")
    .slice(0, 80);

  let candidate = base || `section-${index + 1}`;
  let serial = 2;

  while (used.has(candidate)) {
    candidate = `${base || `section-${index + 1}`}-${serial}`;
    serial += 1;
  }

  used.add(candidate);
  return candidate;
}

function buildNoteToc() {
  els.noteToc.innerHTML = "";
  const headings = [...els.noteContent.querySelectorAll("h1, h2, h3, h4, h5, h6")];

  if (!headings.length) {
    els.noteStructure.hidden = true;
    return;
  }

  const usedIds = new Set();

  headings.forEach((heading, index) => {
    if (!heading.id) {
      heading.id = slugifyHeading(heading.textContent || "", index, usedIds);
    } else {
      usedIds.add(heading.id);
    }

    const level = Number(heading.tagName[1]);
    const li = document.createElement("li");
    const button = document.createElement("button");

    button.type = "button";
    button.className = `toc-link tap-bounce toc-l${level}`;
    button.textContent = heading.textContent || "";

    button.addEventListener("click", () => {
      heading.scrollIntoView({ behavior: "smooth", block: "start" });
    });

    li.appendChild(button);
    els.noteToc.appendChild(li);
  });

  els.noteStructure.hidden = false;
}

async function renderMath(container) {
  if (!window.MathJax || typeof window.MathJax.typesetPromise !== "function") return;
  try {
    await window.MathJax.typesetPromise([container]);
  } catch (error) {
    console.error(error);
  }
}

async function renderActiveNote() {
  const note = state.notes.find((item) => item.id === state.activeNoteId);

  if (!note) {
    els.noteTitle.textContent = getText("noteEmpty");
    els.noteContent.innerHTML = "";
    els.noteSearchWrap.hidden = true;
    els.noteSearchResult.textContent = "";
    els.noteStructure.hidden = true;
    return;
  }

  els.noteTitle.textContent = getLocalizedValue(note.title);
  els.noteContent.innerHTML = markdownToHtml(note.raw);
  els.noteSearchWrap.hidden = false;

  buildNoteToc();
  await renderMath(els.noteContent);

  if (state.noteQuery.trim()) {
    const count = highlightInElement(els.noteContent, state.noteQuery);
    els.noteSearchResult.textContent = count
      ? getText("searchNoteResult", { count })
      : getText("searchNone");
  } else {
    els.noteSearchResult.textContent = "";
  }
}

function getPostExcerpt(content) {
  const plain = stripMarkdown(content);
  const max = 54;
  return plain.length > max ? `${plain.slice(0, max)}...` : plain;
}

async function renderBlog() {
  els.blogList.innerHTML = "";

  state.blogs
    .slice()
    .sort((a, b) => (a.date < b.date ? 1 : -1))
    .forEach((post) => {
      const localizedTitle = getLocalizedValue(post.title) || getText("untitledPost");
      const content = post.localizedContent || "";

      const card = document.createElement("article");
      card.className = "card blog-card";

      const body = document.createElement("div");
      body.className = "blog-body";

      const h3 = document.createElement("h3");
      h3.textContent = localizedTitle;

      const date = document.createElement("p");
      date.className = "small";
      date.textContent = post.date;

      const summary = document.createElement("div");
      summary.className = "blog-summary";

      if (post.image) {
        const thumb = document.createElement("img");
        thumb.className = "blog-thumb";
        thumb.src = post.image;
        thumb.alt = localizedTitle;
        summary.appendChild(thumb);
      }

      const preview = document.createElement("p");
      preview.textContent = getPostExcerpt(content);
      preview.className = "blog-preview";
      summary.appendChild(preview);

      const toggle = document.createElement("button");
      toggle.className = "tap-bounce";
      toggle.textContent = getText("readMore");

      const detail = document.createElement("div");
      detail.className = "blog-content markdown";
      detail.innerHTML = markdownToHtml(content);

      if (post.image) {
        const detailImage = document.createElement("img");
        detailImage.className = "blog-detail-image";
        detailImage.src = post.image;
        detailImage.alt = localizedTitle;
        detail.prepend(detailImage);
      }

      toggle.addEventListener("click", async () => {
        const visible = detail.classList.toggle("show");
        card.classList.toggle("expanded", visible);
        toggle.textContent = visible ? getText("readLess") : getText("readMore");
        summary.hidden = visible;
        summary.classList.toggle("is-hidden", visible);
        summary.style.display = visible ? "none" : "";
        if (visible) {
          await renderMath(detail);
        }
      });

      body.append(h3, date, summary, toggle, detail);
      card.appendChild(body);
      els.blogList.appendChild(card);
    });
}

function bindEvents() {
  els.nav.addEventListener("click", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLButtonElement)) return;

    const page = target.getAttribute("data-page");
    if (!page) return;

    document.querySelectorAll(".nav-btn").forEach((btn) => btn.classList.remove("active"));
    target.classList.add("active");

    els.pages.forEach((section) => {
      section.classList.toggle("active", section.id === `${page}-page`);
    });
  });

  els.globalSearchButton.addEventListener("click", () => {
    state.globalQuery = els.globalSearchInput.value;
    renderNotesList();
  });

  els.globalSearchInput.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      state.globalQuery = els.globalSearchInput.value;
      renderNotesList();
    }
  });

  els.noteSearchButton.addEventListener("click", async () => {
    state.noteQuery = els.noteSearchInput.value;
    await renderActiveNote();
  });

  els.noteSearchInput.addEventListener("keydown", async (event) => {
    if (event.key === "Enter") {
      state.noteQuery = els.noteSearchInput.value;
      await renderActiveNote();
    }
  });
}

function initializeLocale() {
  const preferred = localStorage.getItem("preferred-locale");
  const availableCodes = new Set(enabledLanguages.map((lang) => lang.code));

  if (preferred && availableCodes.has(preferred)) {
    state.locale = preferred;
    return;
  }

  if (!availableCodes.has(state.locale)) {
    state.locale = enabledLanguages[0]?.code || DEFAULT_LOCALE;
  }
}

async function refreshAllViews() {
  applyI18n();
  renderLanguageSwitcher();
  renderNotesList();
  await renderActiveNote();
  await renderBlog();
}

async function init() {
  initializeLocale();
  setProfile();
  bindEvents();
  startSakura();

  await reloadLocalizedContent();
  await refreshAllViews();
}

init().catch((error) => {
  console.error(error);
  els.noteTitle.textContent = "Failed to initialize site.";
});
