// 多语言内容对象（这里只举例说明，你可以扩展更多页面内容）
const translations = {
  en: {
    intro: "Hi, I'm Xizuo! 🌸",
    desc: "A fintech student passionate about coding, data and finance.\nExploring the world with Python, C++, and curiosity.",
    placeholder: "Search..."
  },
  zh: {
    intro: "你好，我是 Xizuo！🌸",
    desc: "一名热爱编程、数据与金融科技的学生。\n正在用 Python、C++ 和好奇心探索世界。",
    placeholder: "搜索..."
  },
  tw: {
    intro: "你好，我是 Xizuo！🌸",
    desc: "一位熱愛程式、資料與金融科技的學生。\n使用 Python、C++ 和好奇心探索世界。",
    placeholder: "搜尋..."
  },
  jp: {
    intro: "こんにちは、Xizuo です！🌸",
    desc: "プログラミング、データ、フィンテックに情熱を持つ学生。\nPythonとC++で世界を探検中。",
    placeholder: "検索..."
  },
  fr: {
    intro: "Bonjour, je suis Xizuo ! 🌸",
    desc: "Étudiante en fintech, passionnée par le code, les données et la finance.\nJ'explore le monde avec Python, C++ et curiosité.",
    placeholder: "Rechercher..."
  }
};

document.addEventListener("DOMContentLoaded", () => {
  const langSelect = document.getElementById("language-select");
  const introText = document.getElementById("intro-text");
  const introDesc = document.getElementById("intro-desc");
  const searchInput = document.getElementById("search");

  langSelect.addEventListener("change", () => {
    const lang = langSelect.value;
    if (translations[lang]) {
      introText.textContent = translations[lang].intro;
      introDesc.textContent = translations[lang].desc;
      searchInput.placeholder = translations[lang].placeholder;
    }
  });

  // 🔍 简单搜索功能（示意）
  searchInput.addEventListener("input", (e) => {
    const query = e.target.value.toLowerCase();
    console.log("Searching:", query); // 你可以自定义搜索行为
  });
});
