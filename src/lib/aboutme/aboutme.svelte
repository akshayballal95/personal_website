<script>
    import img from '$lib/assets/img.png';
    import resume from '$lib/assets/projects/resume.pdf';
    import { goto } from '$app/navigation';
</script>

<div class="about-wrap">
    <!-- Left: profile card -->
    <aside class="profile-card">
        <div class="avatar-ring">
            <img src={img} alt="Akshay Ballal" class="avatar-img" />
        </div>

        <div class="profile-meta">
            <h1 class="profile-name">Akshay Ballal</h1>
            <div class="name-divider"></div>
            <p class="profile-role">MACHINE LEARNING ENGINEER</p>
        </div>

        <div class="social-row">
            <a class="social-btn" href="https://twitter.com/akshayballal95" target="_blank" rel="noopener noreferrer" aria-label="Twitter">
                <i class="fa-brands fa-twitter" />
            </a>
            <a class="social-btn" href="https://www.facebook.com/arballal/" target="_blank" rel="noopener noreferrer" aria-label="Facebook">
                <i class="fa-brands fa-facebook" />
            </a>
            <a class="social-btn" href="https://github.com/akshayballal95" target="_blank" rel="noopener noreferrer" aria-label="GitHub">
                <i class="fa-brands fa-github" />
            </a>
            <a class="social-btn" href="https://www.linkedin.com/in/akshay-ballal/" target="_blank" rel="noopener noreferrer" aria-label="LinkedIn">
                <i class="fa-brands fa-linkedin" />
            </a>
        </div>
    </aside>

    <!-- Right: intro -->
    <div class="intro">
        <div class="intro-heading-block">
            <p class="eyebrow">Hello</p>
            <h2 class="intro-heading">Here's who I am<br />and what I do</h2>
        </div>

        <div class="cta-row">
            <a href={resume} download>
                <button type="button" class="btn-amber">
                    <i class="fa-regular fa-download" />
                    <span>Resume</span>
                </button>
            </a>
            <button type="button" class="btn-ghost" on:click={() => goto('/chat')}>
                Chat With My AI Assistant
            </button>
        </div>

        <p class="bio">
            Welcome to my personal website! I am an AI developer pursuing a Master's in Artificial
            Intelligence at TU Eindhoven, where I am constantly applying First Principles thinking to
            solve complex problems. I am the creator of EmbedAnything, a Rust-based library designed to
            optimize embedding pipelines, making them faster and more reliable. As the head of the
            Machine Learning team at Team Serpentine, I am leading the development of AI solutions for
            Brain-Computer Interfaces. With a passion for advancing technology, I am committed to using
            AI for social good and exploring green technology innovations.
        </p>
    </div>
</div>

<style>
.about-wrap {
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

@media (min-width: 768px) {
    .about-wrap {
        flex-direction: row;
        align-items: flex-start;
        gap: 2.5rem;
    }
}

/* ── Profile card ── */
.profile-card {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 1.25rem;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(159, 107, 35, 0.22);
    border-radius: 18px;
    padding: 2rem 1.75rem 1.5rem;
    flex-shrink: 0;
    width: 100%;
    max-width: 270px;
    align-self: flex-start;
}

@media (max-width: 767px) {
    .profile-card { max-width: 100%; }
}

/* ── Avatar ── */
.avatar-ring {
    position: relative;
    width: 128px;
    height: 128px;
}

.avatar-ring::before {
    content: '';
    position: absolute;
    inset: -2px;
    border-radius: 50%;
    background: conic-gradient(from 180deg at 50% 50%, #9f6b23 0%, #f6c636 40%, #D4900A 70%, #9f6b23 100%);
    z-index: 0;
}

.avatar-img {
    position: relative;
    z-index: 1;
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 50%;
    border: 3px solid #0e0e10;
}

/* ── Profile meta ── */
.profile-meta {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.65rem;
    text-align: center;
}

.profile-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.35rem;
    font-weight: 700;
    letter-spacing: -0.025em;
    color: #F5F0EB;
    margin: 0;
    line-height: 1;
}

.name-divider {
    width: 2rem;
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #9f6b23, transparent);
    border-radius: 1px;
}

.profile-role {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.58rem;
    font-weight: 400;
    letter-spacing: 0.2em;
    color: rgba(159, 107, 35, 0.9);
    text-transform: uppercase;
    margin: 0;
}

/* ── Social ── */
.social-row {
    display: flex;
    gap: 0.4rem;
    padding-top: 0.75rem;
    border-top: 1px solid rgba(255, 255, 255, 0.07);
    width: 100%;
    justify-content: center;
}

.social-btn {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    border-radius: 8px;
    color: rgba(245, 240, 235, 0.45);
    text-decoration: none;
    transition: color 0.2s ease, background 0.2s ease;
    font-size: 1rem;
}

.social-btn:hover {
    color: #D4900A;
    background: rgba(159, 107, 35, 0.12);
}

/* ── Introduction ── */
.intro {
    display: flex;
    flex-direction: column;
    gap: 1.75rem;
    justify-content: center;
}

.intro-heading-block {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    color: #9f6b23;
    text-transform: uppercase;
    margin: 0;
}

.intro-heading {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: clamp(1.6rem, 3vw, 2.4rem);
    font-weight: 700;
    letter-spacing: -0.03em;
    line-height: 1.15;
    color: #F5F0EB;
    margin: 0;
}

/* ── Buttons ── */
.cta-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.btn-amber {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1.4rem;
    background: #9f6b23;
    color: #FFF8F0;
    border: none;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 500;
    cursor: pointer;
    transition: background 0.2s ease, transform 0.15s ease, box-shadow 0.2s ease;
    white-space: nowrap;
}

.btn-amber:hover {
    background: #B87A2A;
    transform: translateY(-1px);
    box-shadow: 0 6px 24px rgba(159, 107, 35, 0.28);
}

.btn-ghost {
    display: inline-flex;
    align-items: center;
    padding: 0.6rem 1.4rem;
    background: transparent;
    color: rgba(245, 240, 235, 0.65);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 400;
    cursor: pointer;
    transition: border-color 0.2s ease, color 0.2s ease, transform 0.15s ease;
    white-space: nowrap;
}

.btn-ghost:hover {
    border-color: rgba(159, 107, 35, 0.4);
    color: rgba(245, 240, 235, 0.92);
    transform: translateY(-1px);
}

/* ── Bio ── */
.bio {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 300;
    line-height: 1.85;
    color: rgba(245, 240, 235, 0.6);
    margin: 0;
}
</style>
