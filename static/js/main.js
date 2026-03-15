/* ============================================================
   Smart Attendance System — Main JavaScript
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {
  // ---------- Flash message auto-dismiss ----------
  const flashMessages = document.querySelectorAll(".flash-message");
  flashMessages.forEach((msg) => {
    setTimeout(() => {
      msg.style.transition = "opacity 0.4s, transform 0.4s";
      msg.style.opacity = "0";
      msg.style.transform = "translateY(-8px)";
      setTimeout(() => msg.remove(), 400);
    }, 4000);
  });

  // ---------- Mobile sidebar toggle ----------
  const mobileToggle = document.getElementById("mobile-toggle");
  const sidebar = document.getElementById("sidebar");

  if (mobileToggle && sidebar) {
    mobileToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });

    // Close sidebar when clicking outside on mobile
    document.addEventListener("click", (e) => {
      if (
        sidebar.classList.contains("open") &&
        !sidebar.contains(e.target) &&
        !mobileToggle.contains(e.target)
      ) {
        sidebar.classList.remove("open");
      }
    });
  }

  // ---------- File upload preview ----------
  const fileInput = document.getElementById("photo-input");
  const previewContainer = document.getElementById("file-preview");
  const uploadZone = document.getElementById("upload-zone");

  if (fileInput && previewContainer) {
    fileInput.addEventListener("change", () => {
      previewContainer.innerHTML = "";
      const files = fileInput.files;

      if (files.length > 0) {
        for (let i = 0; i < files.length; i++) {
          const item = document.createElement("span");
          item.className = "file-preview-item";
          item.textContent = files[i].name;
          previewContainer.appendChild(item);
        }
      }
    });
  }

  if (uploadZone) {
    uploadZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      uploadZone.classList.add("dragover");
    });

    uploadZone.addEventListener("dragleave", () => {
      uploadZone.classList.remove("dragover");
    });

    uploadZone.addEventListener("drop", (e) => {
      e.preventDefault();
      uploadZone.classList.remove("dragover");
      if (fileInput && e.dataTransfer.files.length > 0) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event("change"));
      }
    });
  }

  // ---------- Date filter (history page) ----------
  const dateInput = document.getElementById("date-filter");
  if (dateInput) {
    dateInput.addEventListener("change", () => {
      const date = dateInput.value;
      if (date) {
        window.location.href = `/history?date=${date}`;
      }
    });
  }

  // ---------- Delete confirmation modal ----------
  const modal = document.getElementById("delete-modal");
  const modalName = document.getElementById("modal-person-name");
  const modalForm = document.getElementById("modal-delete-form");
  const modalCancel = document.getElementById("modal-cancel");

  window.confirmDelete = function (name) {
    if (modal && modalName && modalForm) {
      modalName.textContent = name;
      modalForm.action = `/api/delete-person/${encodeURIComponent(name)}`;
      modal.classList.add("active");
    }
  };

  if (modalCancel && modal) {
    modalCancel.addEventListener("click", () => {
      modal.classList.remove("active");
    });
  }

  if (modal) {
    modal.addEventListener("click", (e) => {
      if (e.target === modal) {
        modal.classList.remove("active");
      }
    });
  }

  // ---------- CSV Export ----------
  window.exportCSV = function (date) {
    if (date) {
      window.location.href = `/api/export/${date}`;
    }
  };
});
