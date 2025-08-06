// Custom JavaScript for InterpolatePy documentation

// MathJax configuration
window.MathJax = {
    tex: {
        inlineMath: [["\\(", "\\)"]],
        displayMath: [["\\[", "\\]"]],
        processEscapes: true,
        processEnvironments: true
    },
    options: {
        ignoreHtmlClass: ".*|",
        processHtmlClass: "arithmatex"
    }
};

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", function() {
    // Add copy buttons to code blocks
    const codeBlocks = document.querySelectorAll('pre > code');
    codeBlocks.forEach(function(codeBlock) {
        if (codeBlock.parentNode.querySelector('.copy-button')) return;
        
        const button = document.createElement('button');
        button.className = 'copy-button';
        button.textContent = 'Copy';
        button.addEventListener('click', function() {
            navigator.clipboard.writeText(codeBlock.textContent).then(function() {
                button.textContent = 'Copied!';
                setTimeout(function() {
                    button.textContent = 'Copy';
                }, 2000);
            });
        });
        
        codeBlock.parentNode.style.position = 'relative';
        codeBlock.parentNode.appendChild(button);
    });

    // Smooth scrolling for anchor links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth',
                    block: 'start'
                });
            }
        });
    });

    // Add performance badges to algorithm sections
    const algorithmSections = document.querySelectorAll('h3, h4');
    algorithmSections.forEach(function(heading) {
        const text = heading.textContent.toLowerCase();
        let badge = null;
        
        if (text.includes('cubic spline') || text.includes('linear')) {
            badge = createPerformanceBadge('Fast', 'performance-fast');
        } else if (text.includes('b-spline') || text.includes('smoothing')) {
            badge = createPerformanceBadge('Medium', 'performance-medium');
        } else if (text.includes('optimization') || text.includes('search')) {
            badge = createPerformanceBadge('Slow', 'performance-slow');
        }
        
        if (badge) {
            heading.appendChild(badge);
        }
    });
});

function createPerformanceBadge(text, className) {
    const badge = document.createElement('span');
    badge.className = `performance-badge ${className}`;
    badge.textContent = text;
    return badge;
}