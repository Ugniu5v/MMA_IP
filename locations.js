let ALL_LOCATIONS = [];

document.addEventListener("DOMContentLoaded", async () => {
  const input = document.getElementById("location");
  const dropdown = document.getElementById("locationDropdown");

  try {
    const res = await fetch("http://127.0.0.1:8000/locations");
    ALL_LOCATIONS = await res.json();
  } catch (e) {
    console.error("Failed to load locations", e);
  }

  function renderList(filterText = "") {
    dropdown.innerHTML = "";

    const query = filterText.toLowerCase();

    const matches = ALL_LOCATIONS.filter(loc =>
      loc.toLowerCase().includes(query)
    );

    if (matches.length === 0) {
      dropdown.classList.add("d-none");
      return;
    }

    matches.forEach(loc => {
      const item = document.createElement("button");
      item.type = "button";
      item.className = "list-group-item list-group-item-action";
      item.textContent = loc;

      item.addEventListener("click", () => {
        input.value = loc;
        dropdown.classList.add("d-none");
      });

      dropdown.appendChild(item);
    });

    dropdown.classList.remove("d-none");
  }

  input.addEventListener("input", () => {
    renderList(input.value);
  });
  input.addEventListener("focus", () => {
    renderList(input.value);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest("#location")) {
      dropdown.classList.add("d-none");
    }
  });
});
