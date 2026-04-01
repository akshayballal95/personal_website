<footer class="site-footer">
    <div class="footer-inner">
        <p class="footer-copy">© 2024 Akshay Ballal</p>

        <div class="footer-contacts">
            <a href="tel:+310645791146" class="footer-contact">
                <i class="fa-solid fa-phone footer-icon" />
                <span>+31 0645791146</span>
            </a>
            <a href="mailto:arballal95@gmail.com" class="footer-contact">
                <i class="fa-solid fa-envelope footer-icon" />
                <span>arballal95@gmail.com</span>
            </a>
        </div>

        <div class="footer-social">
            <a class="footer-social-btn" href="https://twitter.com/akshayballal95" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
                <i class="fa-brands fa-twitter" />
            </a>
            <a class="footer-social-btn" href="https://www.facebook.com/arballal/" target="_blank" rel="noopener noreferrer" aria-label="Facebook">
                <i class="fa-brands fa-facebook" />
            </a>
            <a class="footer-social-btn" href="https://github.com/akshayballal95" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <i class="fa-brands fa-github" />
            </a>
            <a class="footer-social-btn" href="https://www.linkedin.com/in/akshay-ballal/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                <i class="fa-brands fa-linkedin" />
            </a>
        </div>
    </div>
</footer>

<style>
.site-footer {
    border-top: 1px solid var(--border-subtle);
    padding: 1rem 2rem;
}

.footer-inner {
    max-width: 1440px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    align-items: center;
    text-align: center;
}

@media (min-width: 768px) {
    .footer-inner {
        flex-direction: row;
        justify-content: space-between;
        text-align: left;
        gap: 1.5rem;
    }
}

.footer-copy {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    color: var(--text-faint);
    margin: 0;
    white-space: nowrap;
}

.footer-contacts {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    align-items: center;
}

@media (min-width: 768px) {
    .footer-contacts {
        flex-direction: row;
        gap: 1.5rem;
        align-items: center;
    }
}

.footer-contact {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    color: var(--text-faint);
    text-decoration: none;
    transition: color 0.2s ease;
    white-space: nowrap;
}

.footer-contact:hover {
    color: var(--text-muted);
}

.footer-icon {
    font-size: 0.65rem;
    color: rgba(159, 107, 35, 0.6);
}

.footer-social {
    display: flex;
    gap: 0.25rem;
}

.footer-social-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 30px;
    height: 30px;
    border-radius: 6px;
    color: var(--text-faint);
    text-decoration: none;
    font-size: 0.85rem;
    transition: color 0.2s ease, background 0.2s ease;
}

.footer-social-btn:hover {
    color: #D4900A;
    background: rgba(159, 107, 35, 0.1);
}
</style>
