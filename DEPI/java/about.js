/*=====================================
        ABOUT PAGE
=====================================*/

document.addEventListener("DOMContentLoaded", () => {

    /*==============================
        Fade In Animation
    ==============================*/

    const elements = document.querySelectorAll(

        ".info-box, .tech-card, .feature-item, .team-card"

    );

    elements.forEach((element, index) => {

        element.style.opacity = "0";
        element.style.transform = "translateY(30px)";
        element.style.transition = "0.6s ease";

        setTimeout(() => {

            element.style.opacity = "1";
            element.style.transform = "translateY(0)";

        }, index * 120);

    });

});


/*==============================
        Smooth Hover
==============================*/

const cards = document.querySelectorAll(

    ".info-box, .tech-card, .feature-item, .team-card"

);

cards.forEach(card => {

    card.addEventListener("mouseenter", () => {

        card.style.transition = "0.3s";

    });

});


/*==============================
        Footer Year
==============================*/

const footer = document.querySelector("footer");

if (footer) {

    const year = new Date().getFullYear();

    footer.innerHTML = `© ${year} AI Medical Center | Graduation Project`;

}