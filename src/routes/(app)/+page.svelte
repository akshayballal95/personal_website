<script lang="ts">
	import Aboutme from '$lib/aboutme/aboutme.svelte';
	import BlogList from '$lib/blogs/blog_list.svelte';
	import { tabSet } from '$lib/stores/stateStore';
	import { onMount } from 'svelte';
	import type { Blog } from '../../input_model';

	/** @type {import('./$types').PageData} */
	export let data: { blogs: Blog[] };
	onMount(() => {
	});


let githubStats = {
	stars: 0,
	forks: 0
};

onMount(async () => {
	$tabSet = 0;

	try {
		const response = await fetch('https://api.github.com/repos/StarlightSearch/EmbedAnything');
		const data = await response.json();
		githubStats = {
			stars: data.stargazers_count,
			forks: data.forks_count
		};
	} catch (error) {
		console.error('Error fetching GitHub stats:', error);
	}
});
</script>

<!-- YOU CAN DELETE EVERYTHING IN THIS PAGE -->

<svelte:head>
	<title>About Me</title>
	<meta
		name="description"
		content="Welcome to my personal website! I am a self-taught AI developer driven by a passion for pushing the boundaries of technology. Applying First Principles thinking, I strive to solve complex challenges and create innovative solutions. As a Technology Enthusiast, I constantly explore the latest advancements in the field. I am deeply committed to leveraging AI for social good and advocate for green technology. Join me on this journey as I utilize my self-taught expertise to build products and technologies that address industrial problems. Let's collaborate and shape a better future together."
	/>
	<meta name="title" content="Akshay Ballal: Machine Learning Enthusiast">
</svelte:head>

<div class="h-full  justify-center items-center p-3 xl:p-12 ">
	<div class="flex xl:flex-row flex-col gap-10 w-full max-w-7xl">
		<div class="xl:basis-3/5 xl:max-w-3xl" >
			<Aboutme />
		</div>

		<div class="flex flex-col gap-3 xl:basis-2/5">
			<h3>Recent Blogs</h3>
			<BlogList blogs={data.blogs} />
		</div>
	</div>
	
<!-- Projects Section -->
<div class="mt-8">
	<h2 class="mb-6">Featured Projects</h2>
	<div class="flex flex-col lg:flex-row gap-6">
		<!-- EmbedAnything Card -->
		<div class="card p-6 w-full">
			<div class="flex flex-col md:flex-row items-center gap-6">
				<img
					src="https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png"
					alt="EmbedAnything Preview"
					class="w-full md:w-1/3 rounded-lg shadow-lg"
				/>
				<div class="flex-1">
					<div class="flex flex-col gap-3 mb-3">
						<h3 class="h3">EmbedAnything</h3>
						<div class="flex flex-wrap gap-3 items-center text-sm">
							<span class="badge variant-filled-secondary">
								<i class="fab fa-github mr-1" />
								{githubStats.stars} stars
							</span>
							<span class="badge variant-filled-secondary">
								<i class="fa fa-code-fork mr-1" />
								{githubStats.forks} forks
							</span>

							<img src="https://static.pepy.tech/badge/embed-anything" alt="PyPI version" />
						</div>
					</div>
					<p class="mb-4">
						A Rust-based library designed to optimize embedding pipelines, making them faster and
						more reliable. Features include:
					</p>
					<ul class="list-disc ml-5 mb-4 space-y-2 text-sm">
						<li>Local, ONNX, and cloud embedding model support</li>
						<li>Multimodal processing (text, images, audio)</li>
						<li>Vector streaming for memory-efficient indexing</li>
						<li>91.2% Rust, 8.5% Python codebase</li>
					</ul>
					<div class="flex-col gap-3 ">
						<div class="flex flex-col gap-3">
							<a
								href="https://github.com/StarlightSearch/EmbedAnything"
								target="_blank"
								rel="noopener noreferrer"
								class="btn variant-filled-primary"
							>
								<i class="fab fa-github mr-2" /> View on GitHub
							</a>
							<a
								href="https://pypi.org/project/embed-anything/"
								target="_blank"
								rel="noopener noreferrer"
								class="btn variant-filled-tertiary"
							>
								<i class="fab fa-python mr-2" /> View on PyPI
							</a>
						</div>
					</div>
				</div>
			</div>
		</div>

		<!-- Candle Contributions Card -->
		<div class="card p-6 w-full">
			<div class="flex flex-col md:flex-row items-center gap-6">
				<img
					src="https://opengraph.githubassets.com/6f2dae8cda2be80144bbe5568e587a4e94715e52deee9b3c86ff465805d232ea/huggingface/candle"
					alt="Candle Logo"
					class="w-full md:w-1/3 rounded-lg shadow-lg"
				/>
				<div class="flex-1">
					<h3 class="h3 mb-3">Contributor to Candle</h3>
					<p class="mb-4">
						Contributing to Candle, a minimalist ML framework for Rust with 16k+ stars. Added
						three key models to enhance the framework's capabilities:
					</p>
					<ul class="list-disc ml-5 mb-4 space-y-2 text-sm">
						<li>ColPali - Vision RAG Model</li>
						<li>Splade Model - Sparse Bert Model for BM25</li>
						<li>Reranker Model - Cross-encoder Architecture</li>
					</ul>
					<a
						href="https://github.com/huggingface/candle/commits?author=akshayballal95"
						target="_blank"
						rel="noopener noreferrer"
						class="btn variant-filled-primary"
					>
						<i class="fab fa-github mr-2" /> View Contributions
					</a>
				</div>
			</div>
		</div>
	</div>
</div>
	
</div>
