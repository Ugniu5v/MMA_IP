let allTitles = [];

document.addEventListener("DOMContentLoaded", async () => {
  const input = document.getElementById("title");
  const dropdown = document.getElementById("titleDropdown");

  const renderDropdown = (items) => {
    dropdown.innerHTML = "";

    items.forEach(title => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "dropdown-item";
      item.textContent = title;

      item.onclick = () => {
        input.value = title;
        dropdown.style.display = "none";
      };

      dropdown.appendChild(item);
    });

    dropdown.style.display = items.length ? "block" : "none";
  };

  try {
    const res = await fetch("http://127.0.0.1:8000/titles");
    allTitles = await res.json();
  } catch (e) {
    console.error("Failed to load titles", e);
  }

  input.addEventListener("focus", () => {
    renderDropdown(allTitles.slice(0, 50));
  });

  input.addEventListener("input", () => {
    const value = input.value.toLowerCase().trim();

    if (!value) {
      renderDropdown(allTitles.slice(0, 50));
      return;
    }

    const matches = allTitles
      .filter(t => t.toLowerCase().includes(value))
      .slice(0, 50);

    renderDropdown(matches);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest("#title") && !e.target.closest("#titleDropdown")) {
      dropdown.style.display = "none";
    }
  });
});

