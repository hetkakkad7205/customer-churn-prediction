
document.addEventListener("DOMContentLoaded", function () {
    const form = document.getElementById("predictionForm");

    if (form) {
        form.addEventListener("submit", function (event) {
            if (!form.checkValidity()) {
                event.preventDefault();
                form.reportValidity();
                return;
            }

            const button = form.querySelector('button[type="submit"]');

            if (button) {
                button.disabled = true;
                button.textContent = "Predicting...";
            }
        });
    }
});
