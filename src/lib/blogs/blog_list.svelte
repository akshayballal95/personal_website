<script lang="ts">
    import type { Blog } from '../../input_model';
    export let blogs: Blog[];

    function formattedDate(date: Date) {
        return date.toLocaleDateString('en-GB', {
            day: 'numeric', month: 'short', year: 'numeric'
        });
    }
</script>

<div class="blog-list">
    {#each blogs as blog}
        <a href="/blogs/{blog.id}" class="blog-card">
            <div class="blog-img-wrap">
                <img class="blog-img" src={blog.image} alt="" />
            </div>
            <div class="blog-body">
                <span class="blog-date">{formattedDate(blog.date)}</span>
                <h3 class="blog-title">{blog.title}</h3>
                <p class="blog-desc">{blog.description}</p>
            </div>
        </a>
    {/each}
</div>

<style>
.blog-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    width: 100%;
}

.blog-card {
    display: flex;
    gap: 1rem;
    padding: 0.875rem 0.875rem 1rem;
    background: rgba(255, 255, 255, 0.025);
    border: 1px solid rgba(255, 255, 255, 0.065);
    border-radius: 12px;
    text-decoration: none !important;
    transition: background 0.22s ease, border-color 0.22s ease, transform 0.22s ease;
    align-items: flex-start;
}

.blog-card:hover {
    background: rgba(159, 107, 35, 0.055);
    border-color: rgba(159, 107, 35, 0.28);
    transform: translateX(3px);
}

.blog-img-wrap {
    flex-shrink: 0;
    width: 68px;
    height: 68px;
    border-radius: 8px;
    overflow: hidden;
    background: rgba(255,255,255,0.04);
}

.blog-img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 8px;
    transition: transform 0.3s ease;
    display: block;
}

.blog-card:hover .blog-img {
    transform: scale(1.06);
}

.blog-body {
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
    min-width: 0;
}

.blog-date {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 0.06em;
    color: rgba(159, 107, 35, 0.8);
    text-transform: uppercase;
    text-decoration: none;
}

.blog-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.875rem;
    font-weight: 600;
    line-height: 1.55;
    color: #F5F0EB;
    margin: 0;
    transition: color 0.2s ease;
    text-decoration: none;
}

.blog-card:hover .blog-title {
    color: #D4900A;
}

.blog-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.75rem;
    font-weight: 300;
    line-height: 1.6;
    color: rgba(245, 240, 235, 0.42);
    margin: 0;
    padding-bottom: 2px;
    display: -webkit-box;
    -webkit-line-clamp: 3;
    -webkit-box-orient: vertical;
    overflow: hidden;
    text-decoration: none;
}
</style>
