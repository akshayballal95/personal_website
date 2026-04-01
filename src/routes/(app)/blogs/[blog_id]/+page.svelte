<script lang="ts">
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';

	/** @type {import('./$types').PageData} */
	export let data: { id: string; date: Date; title: string; html: string; description: string; image_url: string };

	function formattedDate(date: Date) {
		return date
			.toLocaleDateString('en-GB', {
				day: 'numeric',
				month: 'short',
				year: 'numeric'
			})
			.replace(/ /g, ' ');
	}
	let y = 0;
	onMount(() => {
		$tabSet = 4;
		document.body.querySelector('#blog')?.scrollTo(0, 0);
	});
</script>

<svelte:head>
	<title>{data.title}</title>
	<meta name="description" content={data.description} />
	<meta name="title" content={data.title} />
	<meta
		property="og:image"
		content={data.image_url}
	/>
	<meta
		property="og:description"
		content={data.description}/>
	<meta property="og:title" content={data.title}/>
	<meta property="og:description" content={data.description} />
	

	<meta name="twitter:card" content="summary_large_image" />
	<meta name="twitter:creator" content="@akshayballal95" />
	<meta property="twitter:image"
		content={data.image_url} />
	<meta property="twitter:title" content={data.title} />
	<meta property="twitter:description"
		content={data.description}/>

		<meta name="robots" content="follow, index, max-snippet:-1, max-video-preview:-1, max-image-preview:large" />
		</svelte:head
>

<svelte:window bind:scrollY={y} />

<div class="flex justify-center lg:p-10 p-5 mx-auto overflow-hidden">
	<div id="blog" class="blog-container flex flex-col gap-8 xl:w-2/3 md:w-4/5 w-full rounded-md card md:p-8 p-5 relative">
		<h2 class="mt-5">{data.title}</h2>
		<p>{formattedDate(data.date)}</p>

		<div class="blog-content">
			{@html data.html}
		</div>
	</div>
</div>

<style>
	/* ── Blog header ── */
	.blog-container h2 {
		font-family: 'Plus Jakarta Sans', sans-serif;
		font-weight: 700;
		line-height: 1.3;
	}
	.blog-container p {
		font-family: 'Plus Jakarta Sans', sans-serif;
	}

	/* ── VS Code–style code blocks ── */
	:global(.blog-container pre.shiki) {
		position: relative;
		margin: 1.5rem 0;
		border-radius: 8px;
		border: 1px solid rgba(255, 255, 255, 0.08);
		border-top: 2px solid rgba(159, 107, 35, 0.5);
		overflow: hidden;
		box-shadow: 0 4px 24px rgba(0, 0, 0, 0.5);
	}

	/* Code area */
	:global(.blog-container pre.shiki code) {
		display: block;
		padding: 1.25rem 1.5rem;
		overflow-x: auto;
		font-family: 'JetBrains Mono', Consolas, 'Courier New', monospace;
		font-size: 0.82rem;
		line-height: 1.7;
		font-weight: 400;
		tab-size: 4;
		-webkit-overflow-scrolling: touch;
	}

	/* Scrollbar */
	:global(.blog-container pre.shiki code::-webkit-scrollbar) {
		height: 4px;
	}
	:global(.blog-container pre.shiki code::-webkit-scrollbar-track) {
		background: transparent;
	}
	:global(.blog-container pre.shiki code::-webkit-scrollbar-thumb) {
		background: rgba(255, 255, 255, 0.12);
		border-radius: 2px;
	}

	/* ── Content typography ── */
	:global(.blog-content > *) {
		margin-bottom: 0.875rem;
	}
	:global(.blog-content > *:last-child) {
		margin-bottom: 0;
	}
	:global(.blog-content p) {
		font-family: 'Plus Jakarta Sans', sans-serif;
		font-weight: 300;
		line-height: 1.85;
		font-size: 0.95rem;
	}
	:global(.blog-content h1),
	:global(.blog-content h2),
	:global(.blog-content h3),
	:global(.blog-content h4) {
		font-family: 'Plus Jakarta Sans', sans-serif;
		font-weight: 700;
		margin-top: 1.75rem;
		margin-bottom: 0.5rem;
		line-height: 1.3;
	}
	:global(.blog-content li) {
		font-family: 'Plus Jakarta Sans', sans-serif;
		font-weight: 300;
		font-size: 0.9rem;
		line-height: 1.75;
	}

	/* Inline code (not inside pre) */
	:global(.blog-container :not(pre) > code) {
		font-family: 'JetBrains Mono', Consolas, monospace;
		font-size: 0.8em;
		padding: 0.15em 0.45em;
		border-radius: 4px;
		background: rgba(255, 255, 255, 0.08);
		color: #ce9178;
		font-weight: 400;
	}
</style>
