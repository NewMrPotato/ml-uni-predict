document.addEventListener("DOMContentLoaded", () => {
  const searchLink = document.querySelector('link[rel="search"]');
  if (!searchLink) return;

  const root = new URL("./", searchLink.href);
  const current = new URL(window.location.href);
  let relativePath = current.pathname.slice(root.pathname.length);
  if (!relativePath || relativePath.endsWith("/")) relativePath += "index.html";

  const isRussian = relativePath.startsWith("ru/");
  const englishPath = isRussian ? relativePath.slice(3) : relativePath;
  const translatedPages = new Set([
    "index.html",
    "getting-started.html",
    "concepts.html",
    "security.html",
    "changelog.html",
    "contributing.html",
    "release-notes-0.2.0a1.html",
    "release-checklist.html",
    "guides/input-data.html",
    "guides/preprocessing.html",
    "guides/classification.html",
    "guides/frameworks.html",
    "guides/model-loading.html",
    "guides/engines.html",
    "guides/ensembles.html",
    "guides/errors.html",
    "api/index.html",
    "api/predictor.html",
    "api/schema.html",
    "api/result.html",
    "api/ensemble.html",
    "api/engines.html",
    "api/loading.html",
    "api/preprocessing.html",
    "api/exceptions.html",
  ]);

  const wrapper = document.createElement("div");
  wrapper.className = "language-switcher";

  const label = document.createElement("label");
  label.htmlFor = "documentation-language";
  label.textContent = isRussian ? "Язык" : "Language";

  const select = document.createElement("select");
  select.id = "documentation-language";
  select.setAttribute("aria-label", isRussian ? "Выбрать язык документации" : "Select documentation language");
  select.innerHTML = '<option value="en">English</option><option value="ru">Русский</option>';
  select.value = isRussian ? "ru" : "en";

  select.addEventListener("change", () => {
    let targetPath = englishPath;
    if (select.value === "ru") {
      targetPath = translatedPages.has(englishPath) ? `ru/${englishPath}` : "ru/index.html";
    }
    window.location.assign(new URL(targetPath, root).href);
  });

  wrapper.append(label, select);
  const searchContainer = document.querySelector(".sidebar-search-container");
  const sidebar = document.querySelector(".sidebar-sticky");
  if (searchContainer) searchContainer.before(wrapper);
  else if (sidebar) sidebar.prepend(wrapper);

  const navigation = document.querySelector(".sidebar-tree");
  const topLevel = navigation?.querySelector(":scope > ul");
  if (!navigation || !topLevel) return;

  const russianRoot = Array.from(topLevel.children).find((item) => {
    const link = item.querySelector(":scope > a");
    return link && new URL(link.href).pathname.startsWith(`${root.pathname}ru/`);
  });

  if (isRussian && russianRoot) {
    const russianNavigation = russianRoot.querySelector(":scope > ul");
    if (russianNavigation) {
      russianNavigation.querySelectorAll("li").forEach((item) => {
        if (item.classList.contains("toctree-l2")) {
          item.classList.replace("toctree-l2", "toctree-l1");
        } else if (item.classList.contains("toctree-l3")) {
          item.classList.replace("toctree-l3", "toctree-l2");
        }
      });
      navigation.replaceChildren(russianNavigation);
    }
    const searchInput = document.querySelector(".sidebar-search");
    if (searchInput) {
      searchInput.placeholder = "Поиск";
      searchInput.setAttribute("aria-label", "Поиск");
    }
  } else if (russianRoot) {
    russianRoot.remove();
  }
});
