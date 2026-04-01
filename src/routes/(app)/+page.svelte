<script lang="ts">
	import Aboutme from '$lib/aboutme/aboutme.svelte';
	import BlogList from '$lib/blogs/blog_list.svelte';
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';
	import type { Blog } from '../../input_model';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };

	let githubStats = { stars: 0, forks: 0 };

	onMount(async () => {
		$tabSet = 0;
		try {
			const response = await fetch('https://api.github.com/repos/StarlightSearch/EmbedAnything');
			const json = await response.json();
			githubStats = { stars: json.stargazers_count, forks: json.forks_count };
		} catch (error) {
			console.error('Error fetching GitHub stats:', error);
		}
	});
</script>

<svelte:head>
	<title>About Me</title>
	<meta name="description" content="Welcome to my personal website! I am a self-taught AI developer driven by a passion for pushing the boundaries of technology." />
	<meta name="title" content="Akshay Ballal: Machine Learning Enthusiast" />
</svelte:head>

<div class="page-wrap">
	<div class="content-grid">
		<!-- About + Blogs row -->
		<div class="top-row">
			<div class="about-col">
				<Aboutme />
			</div>
			<div class="blogs-col">
				<div class="section-label">
					<span class="label-line"></span>
					<h3 class="section-title">Recent Blogs</h3>
				</div>
				<BlogList blogs={data.blogs} />
			</div>
		</div>

		<!-- Featured Projects -->
		<div class="projects-section">
			<div class="section-label">
				<span class="label-line"></span>
				<h3 class="section-title">Featured Projects</h3>
			</div>

			<div class="projects-grid">
				<!-- EmbedAnything -->
				<div class="project-card">
					<div class="project-img-wrap">
						<img
							src="https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png"
							alt="EmbedAnything"
							class="project-img"
						/>
					</div>
					<div class="project-body">
						<div class="project-header">
							<h3 class="project-title">EmbedAnything</h3>
							<div class="project-badges">
								<span class="badge">
									<i class="fab fa-github" />
									{githubStats.stars} stars
								</span>
								<span class="badge">
									<i class="fa fa-code-fork" />
									{githubStats.forks} forks
								</span>
								<img src="https://static.pepy.tech/badge/embed-anything" alt="PyPI downloads" class="badge-img" />
							</div>
						</div>
						<p class="project-desc">
							A Rust-based library designed to optimize embedding pipelines, making them faster and
							more reliable.
						</p>
						<ul class="project-list">
							<li>Local, ONNX, and cloud embedding model support</li>
							<li>Multimodal processing (text, images, audio)</li>
							<li>Vector streaming for memory-efficient indexing</li>
							<li>91.2% Rust, 8.5% Python codebase</li>
						</ul>
						<div class="project-links">
							<a href="https://github.com/StarlightSearch/EmbedAnything" target="_blank" rel="noopener noreferrer" class="project-btn project-btn--primary unstyled">
								<i class="fab fa-github" /> View on GitHub
							</a>
							<a href="https://pypi.org/project/embed-anything/" target="_blank" rel="noopener noreferrer" class="project-btn project-btn--ghost unstyled">
								<i class="fab fa-python" /> PyPI
							</a>
						</div>
					</div>
				</div>

				<!-- Candle Contributions -->
				<div class="project-card">
					<div class="project-img-wrap">
						<img
							src="https://opengraph.githubassets.com/6f2dae8cda2be80144bbe5568e587a4e94715e52deee9b3c86ff465805d232ea/huggingface/candle"
							alt="Candle"
							class="project-img"
						/>
					</div>
					<div class="project-body">
						<div class="project-header">
							<h3 class="project-title">Contributor to Candle</h3>
						</div>
						<p class="project-desc">
							Contributing to Candle, a minimalist ML framework for Rust with 16k+ stars. Added
							three key models to enhance the framework's capabilities.
						</p>
						<ul class="project-list">
							<li>ColPali — Vision RAG Model</li>
							<li>Splade Model — Sparse Bert for BM25</li>
							<li>Reranker Model — Cross-encoder Architecture</li>
						</ul>
						<div class="project-links">
							<a href="https://github.com/huggingface/candle/commits?author=akshayballal95" target="_blank" rel="noopener noreferrer" class="project-btn project-btn--primary unstyled">
								<i class="fab fa-github" /> View Contributions
							</a>
						</div>
					</div>
				</div>
			</div>
		</div>
	</div>
</div>

<style>
.page-wrap {
    padding: 2rem 1.25rem 3rem;
}

@media (min-width: 1024px) {
    .page-wrap { padding: 2.5rem 3rem 4rem; }
}

.content-grid {
    max-width: 1400px;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 3rem;
}

/* ── Top row ── */
.top-row {
    display: flex;
    flex-direction: column;
    gap: 2.5rem;
}

@media (min-width: 1200px) {
    .top-row {
        flex-direction: row;
        gap: 3rem;
    }
    .about-col { flex: 3; min-width: 0; }
    .blogs-col { flex: 2; min-width: 0; }
}

/* ── Section label ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.label-line {
    width: 2px;
    height: 1.1rem;
    background: linear-gradient(180deg, #9f6b23, #D4900A);
    border-radius: 1px;
    flex-shrink: 0;
}

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    color: var(--text-primary);
    margin: 0;
}

/* ── Projects grid ── */
.projects-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 1.25rem;
}

@media (min-width: 900px) {
    .projects-grid { grid-template-columns: 1fr 1fr; }
}

/* ── Project card ── */
.project-card {
    background: var(--surface-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
    transition: border-color 0.25s ease, transform 0.25s ease, box-shadow 0.25s ease;
}

.project-card:hover {
    border-color: rgba(159, 107, 35, 0.3);
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.4);
}

.project-img-wrap {
    width: 100%;
    overflow: hidden;
    background: rgba(128, 128, 128, 0.06);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}

.project-img {
    width: 100%;
    height: auto;
    max-height: 200px;
    object-fit: contain;
    border-radius: 0;
    display: block;
    transition: transform 0.4s ease;
}

.project-card:hover .project-img {
    transform: scale(1.02);
}

.project-body {
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    flex: 1;
}

.project-header {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.project-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    color: var(--text-primary);
    margin: 0;
}

.project-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 0.45rem;
    align-items: center;
}

.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.2rem 0.6rem;
    background: rgba(159, 107, 35, 0.1);
    border: 1px solid rgba(159, 107, 35, 0.22);
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: rgba(212, 144, 10, 0.95);
    letter-spacing: 0.03em;
}

.badge-img {
    height: 20px;
    border-radius: 4px;
}

.project-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.845rem;
    font-weight: 300;
    line-height: 1.7;
    color: var(--text-muted);
    margin: 0;
}

.project-list {
    list-style: none;
    padding: 0;
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.project-list li {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 300;
    color: var(--text-faint);
    padding-left: 1.1rem;
    position: relative;
    margin: 0;
}

.project-list li::before {
    content: '—';
    position: absolute;
    left: 0;
    color: rgba(159, 107, 35, 0.55);
    font-size: 0.65rem;
    top: 1px;
}

.project-links {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6rem;
    margin-top: auto;
    padding-top: 0.5rem;
}

.project-btn {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.5rem 1.1rem;
    border-radius: 8px;
    font-family: 'DM Sans', sans-serif;
    font-size: 0.8rem;
    font-weight: 500;
    text-decoration: none;
    transition: all 0.2s ease;
    white-space: nowrap;
}

.project-btn--primary {
    background: #9f6b23;
    color: #FFF8F0;
    border: 1px solid transparent;
}

.project-btn--primary:hover {
    background: #B87A2A;
    transform: translateY(-1px);
    box-shadow: 0 4px 16px rgba(159, 107, 35, 0.3);
}

.project-btn--ghost {
    background: transparent;
    color: var(--text-muted);
    border: 1px solid var(--border-subtle);
}

.project-btn--ghost:hover {
    border-color: rgba(159, 107, 35, 0.35);
    color: var(--text-primary);
}
</style>
