// ========== 多语言翻译内容 ========== //
const translations = {
  en: {
    // 导航栏
    nav_about: "About",
    nav_blog: "Blog",
    nav_notes: "Notes",
    nav_projects: "Projects",
    nav_faq: "FAQ",

    // 首页 (index.html)
    intro: "Hi, I'm Xizuo! 🌸",
    desc: "A fintech student passionate about coding, data and finance.\nExploring the world with Python, C++, and curiosity.",

    // Blog (blog.html)
    blog_title: "Blog",

    // Notes (notes.html)
    notes_title: "Notes",

    // Projects (projects.html)
    projects_title: "Projects",

    // FAQ (faq.html)
    faq_title: "FAQ",
  },
  zh: {
    nav_about: "关于",
    nav_blog: "博客",
    nav_notes: "笔记",
    nav_projects: "项目",
    nav_faq: "常见问题",

    intro: "你好，我是 Xizuo！🌸",
    desc: "一名热爱编程、数据与金融科技的学生。\n用 Python、C++ 和好奇心探索世界。",

    blog_title: "博客",
    notes_title: "笔记",
    projects_title: "项目",
    faq_title: "常见问题",
  },
  tw: {
    nav_about: "關於",
    nav_blog: "部落格",
    nav_notes: "筆記",
    nav_projects: "項目",
    nav_faq: "常見問題",

    intro: "你好，我是 Xizuo！🌸",
    desc: "一位熱愛程式、資料與金融科技的學生。\n使用 Python、C++ 和好奇心探索世界。",

    blog_title: "部落格",
    notes_title: "筆記",
    projects_title: "項目",
    faq_title: "常見問題",
  },
  jp: {
    nav_about: "紹介",
    nav_blog: "ブログ",
    nav_notes: "ノート",
    nav_projects: "プロジェクト",
    nav_faq: "FAQ",

    intro: "こんにちは、Xizuo です！🌸",
    desc: "プログラミング、データ、フィンテックに情熱を持つ学生。\nPython と C++ で世界を探検中。",

    blog_title: "ブログ",
    notes_title: "ノート",
    projects_title: "プロジェクト",
    faq_title: "よくある質問",
  },
  fr: {
    nav_about: "À propos",
    nav_blog: "Blog",
    nav_notes: "Notes",
    nav_projects: "Projets",
    nav_faq: "FAQ",

    intro: "Bonjour, je suis Xizuo ! 🌸",
    desc: "Étudiante en fintech, passionnée par le code, les données et la finance.\nJ'explore le monde avec Python, C++ et curiosité.",

    blog_title: "Blog",
    notes_title: "Notes",
    projects_title: "Projets",
    faq_title: "FAQ",
  }
};

// ========== 脚本逻辑 ========== //
document.addEventListener("DOMContentLoaded", () => {
  // 找到语言下拉框
  const langSelect = document.getElementById("language-select");
  if (!langSelect) return;

  // 导航栏链接元素
  const navAbout = document.getElementById("nav-about");
  const navBlog = document.getElementById("nav-blog");
  const navNotes = document.getElementById("nav-notes");
  const navProjects = document.getElementById("nav-projects");
  const navFaq = document.getElementById("nav-faq");

  // 各页面特定元素
  const introText = document.getElementById("intro-text");
  const introDesc = document.getElementById("intro-desc");
  const blogTitle = document.getElementById("blog-title");
  const notesTitle = document.getElementById("notes-title");
  const projectsTitle = document.getElementById("projects-title");
  const faqTitle = document.getElementById("faq-title");

  // 监听语言切换事件
  langSelect.addEventListener("change", () => {
    applyTranslations(langSelect.value);
  });

  // 页面加载时，默认使用英文
  applyTranslations("en");

  // ========== 应用翻译的函数 ========== //
  function applyTranslations(lang) {
    if (!translations[lang]) return;

    // 导航栏文字
    if (navAbout)    navAbout.textContent    = translations[lang].nav_about;
    if (navBlog)     navBlog.textContent     = translations[lang].nav_blog;
    if (navNotes)    navNotes.textContent    = translations[lang].nav_notes;
    if (navProjects) navProjects.textContent = translations[lang].nav_projects;
    if (navFaq)      navFaq.textContent      = translations[lang].nav_faq;

    // 首页
    if (introText) introText.textContent = translations[lang].intro;
    if (introDesc) introDesc.textContent = translations[lang].desc;

    // Blog
    if (blogTitle) blogTitle.textContent = translations[lang].blog_title;

    // Notes
    if (notesTitle) notesTitle.textContent = translations[lang].notes_title;

    // Projects
    if (projectsTitle) projectsTitle.textContent = translations[lang].projects_title;

    // FAQ
    if (faqTitle) faqTitle.textContent = translations[lang].faq_title;
  }

  // ========== FAQ：点击展开/收起答案 ========== //
  document.querySelectorAll('.faq-item').forEach(item => {
    item.addEventListener('click', () => {
      const ans = item.querySelector('.faq-answer');
      if (!ans) return;
      ans.classList.toggle('collapsed');
      ans.classList.toggle('expanded');
    });
  });
});
