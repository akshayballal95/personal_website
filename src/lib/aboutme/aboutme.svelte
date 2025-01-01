<script>
	import { Avatar } from '@skeletonlabs/skeleton';
	import img from '$lib/assets/img.png';
	import resume from '$lib/assets/projects/resume.pdf';
	import { goto } from '$app/navigation';
	import { onMount } from 'svelte';

	let githubStats = {
		stars: 0,
		forks: 0
	};

	onMount(async () => {
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

<div class="flex flex-col gap-8">
	<!-- Profile Section -->
	<div class="md:flex-row flex flex-col gap-5 md:p-0 items-center bg-neutral-950 bg-opacity-0">
		<!-- Left Profile Card -->
		<div
			class="flex flex-col items-center gap-5 card md:min-w-[350px] h-full w-full md:w-auto justify-center"
		>
			<Avatar class="mt-10 mb-5" src={img} width="w-40" rounded="rounded-full" />
			<h1 class="unstyled text-4xl">Akshay Ballal</h1>
			<div class="h-0.5 mt-5 w-14 bg-primary-500 rounded-full" />
			<p class="unstyled grow ml-5 mr-5 text-center font-light text-lg tracking-widest">
				MACHINE LEARNING ENTHUSIAST
			</p>
			<div
				class=" h-14 w-full bg-primary-50 flex text-surface-700 justify-around items-center text-3xl self-end"
			>
				<a class="unstyled" href="https://twitter.com/akshayballal95"
					><i class="fa-brands fa-twitter" /></a
				>
				<a class="unstyled" href="https://www.facebook.com/arballal/"
					><i class="fa-brands fa-facebook" /></a
				>
				<a class="unstyled" href="https://github.com/akshayballal95"
					><i class="fa-brands fa-github" /></a
				>
				<a class="unstyled" href="https://www.linkedin.com/in/akshay-ballal/"
					><i class="fa-brands fa-linkedin" /></a
				>
			</div>
		</div>

		<!-- Right Introduction -->
		<div class="flex flex-col justify-center h-full">
			<h2 class="mb-2">Hello</h2>
			<h3 class="mb-10">Here's who I am and what I do</h3>
			<div class="flex flex-col lg:flex-row gap-5 mb-10">
				<a href={resume} download
					><button type="button" class="btn variant-filled rounded-full w-full"
						><i class="fa-regular fa-download" /> <span>Resume</span></button
					></a
				>
				<button
					on:click={() => goto('/chat')}
					type="button"
					class="btn variant-filled-surface rounded-full">Chat With My AI Assistant</button
				>
			</div>
			<p class="text-justify">
				Welcome to my personal website! I am an AI developer pursuing a Master’s in Artificial
				Intelligence at TU Eindhoven, where I am constantly applying First Principles thinking to
				solve complex problems. I am the creator of EmbedAnything, a Rust-based library designed to
				optimize embedding pipelines, making them faster and more reliable. As the head of the
				Machine Learning team at Team Serpentine, I am leading the development of AI solutions for
				Brain-Computer Interfaces. With a passion for advancing technology, I am committed to using
				AI for social good and exploring green technology innovations. Let’s collaborate to shape a
				better future through cutting-edge solutions to real-world challenges.
			</p>
		</div>
	</div>

	<!-- Projects Section -->
	<div class="mt-8">
		<h2 class="mb-6">Featured Projects</h2>
		<div class="flex flex-col gap-6">
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
						<p class="mb-4 text-lg">
							A Rust-based library designed to optimize embedding pipelines, making them faster and
							more reliable. Features include:
						</p>
						<ul class="list-disc ml-5 mb-4 space-y-2">
							<li>Local, ONNX, and cloud embedding model support</li>
							<li>Multimodal processing (text, images, audio)</li>
							<li>Vector streaming for memory-efficient indexing</li>
							<li>91.2% Rust, 8.5% Python codebase</li>
						</ul>
						<div class="flex gap-3">
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
						<p class="mb-4 text-lg">
							Contributing to Candle, a minimalist ML framework for Rust with 16k+ stars. Added
							three key models to enhance the framework's capabilities:
						</p>
						<ul class="list-disc ml-5 mb-4 space-y-2">
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
