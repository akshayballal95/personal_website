<script lang="ts">
    import { tabSet } from '$lib/stores/stateStore';
    import { goto } from '$app/navigation';
    import { drawerStore } from '@skeletonlabs/skeleton';
    import type { DrawerSettings } from '@skeletonlabs/skeleton';

    const links = [
        { label: 'About',    value: 0, href: '/' },
        { label: 'Resume',   value: 1, href: '/resume' },
        { label: 'Projects', value: 2, href: '/projects' },
        { label: 'Contact',  value: 3, href: '/contact' },
        { label: 'Blog',     value: 4, href: '/blogs' },
        { label: 'Chat',     value: 5, href: '/chat' },
    ];

    function openDrawer(): void {
        const s: DrawerSettings = { id: 'demo', position: 'top' };
        drawerStore.open(s);
    }
</script>

<nav class="nav-desktop hidden md:flex" aria-label="Main navigation">
    {#each links as link}
        <button
            class="nav-item"
            class:is-active={$tabSet === link.value}
            on:click={() => { $tabSet = link.value; goto(link.href); }}
        >
            {link.label}
        </button>
    {/each}
</nav>

<button
    class="mobile-toggle md:hidden"
    on:click={openDrawer}
    aria-label="Open navigation"
>
    <span class="bar"></span>
    <span class="bar bar--mid"></span>
    <span class="bar"></span>
</button>

<style>
/* ── Desktop nav ── */
.nav-desktop {
    align-items: center;
    gap: 0;
}

.nav-item {
    position: relative;
    padding: 0.375rem 0.875rem;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 400;
    letter-spacing: 0.01em;
    color: var(--text-muted);
    background: none;
    border: none;
    cursor: pointer;
    transition: color 0.2s ease;
    white-space: nowrap;
}

.nav-item::after {
    content: '';
    position: absolute;
    bottom: -2px;
    left: 50%;
    transform: translateX(-50%) scaleX(0);
    width: 60%;
    height: 1.5px;
    background: linear-gradient(90deg, transparent, #D4900A, transparent);
    border-radius: 1px;
    transition: transform 0.25s cubic-bezier(0.4, 0, 0.2, 1);
}

.nav-item:hover {
    color: var(--text-primary);
}

.nav-item:hover::after {
    transform: translateX(-50%) scaleX(1);
    background: linear-gradient(90deg, transparent, rgba(159, 107, 35, 0.6), transparent);
}

.nav-item.is-active {
    color: #D4900A;
}

.nav-item.is-active::after {
    transform: translateX(-50%) scaleX(1);
    background: linear-gradient(90deg, transparent, #D4900A, transparent);
}

/* ── Mobile toggle ── */
.mobile-toggle {
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 5px;
    width: 36px;
    height: 36px;
    padding: 0;
    background: var(--toggle-bg);
    border: 1px solid var(--toggle-border);
    border-radius: 8px;
    cursor: pointer;
    align-items: center;
    transition: border-color 0.2s ease;
}

.mobile-toggle:hover {
    border-color: rgba(159, 107, 35, 0.4);
}

.bar {
    display: block;
    width: 16px;
    height: 1.5px;
    background: var(--bar-color);
    border-radius: 1px;
    transition: opacity 0.2s ease;
}

.bar--mid {
    width: 11px;
    margin-right: 5px;
}
</style>
