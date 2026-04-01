<script lang="ts">
    import Navlinks from "./navlinks.svelte";
    import { LightSwitch } from '@skeletonlabs/skeleton';
    import { goto } from "$app/navigation";
</script>

<header class="site-header">
    <div class="header-inner">
        <div
            class="header-brand"
            on:click={() => goto("/")}
            on:keydown={(e) => e.key === 'Enter' && goto("/")}
            role="button"
            tabindex="0"
        >
            <span class="brand-name">Akshay Ballal</span>
            <span class="brand-sep">/</span>
            <span class="brand-tag">ML ENGINEER</span>
        </div>
        <div class="header-right">
            <Navlinks />
            <LightSwitch />
        </div>
    </div>
</header>

<style>
.site-header {
    position: sticky;
    top: 0;
    z-index: 100;
    background: rgba(10, 10, 12, 0.82);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border-bottom: 1px solid rgba(159, 107, 35, 0.18);
}

.header-inner {
    max-width: 1440px;
    margin: 0 auto;
    padding: 0 2rem;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.header-brand {
    display: flex;
    align-items: baseline;
    gap: 0.5rem;
    cursor: pointer;
    text-decoration: none;
    outline: none;
}

.brand-name {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 700;
    font-size: 1.2rem;
    letter-spacing: -0.025em;
    color: #F5F0EB;
    transition: opacity 0.2s ease;
}

.header-brand:hover .brand-name {
    opacity: 0.85;
}

.brand-sep {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: rgba(159, 107, 35, 0.5);
    font-size: 1rem;
}

.brand-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    font-weight: 400;
    letter-spacing: 0.18em;
    color: rgba(159, 107, 35, 0.85);
    text-transform: uppercase;
    display: none;
}

@media (min-width: 640px) {
    .brand-tag {
        display: inline;
    }
}

.header-right {
    display: flex;
    align-items: center;
    gap: 1rem;
}
</style>
