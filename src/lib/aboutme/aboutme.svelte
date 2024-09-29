<script>
	import { Avatar } from '@skeletonlabs/skeleton';
	import img from '$lib/assets/img.png';
	import resume from '$lib/assets/projects/resume.pdf';
	import { goto } from '$app/navigation';
</script>

<div class=" md:flex-row flex flex-col gap-5 md:p-0 items-center bg-neutral-950 bg-opacity-0">
	<div
		class=" flex flex-col items-center gap-5 card md:min-w-[350px] h-full w-full md:w-auto justify-center"
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
			optimize embedding pipelines, making them faster and more reliable. As the head of the Machine
			Learning team at Team Serpentine, I am leading the development of AI solutions for
			Brain-Computer Interfaces. With a passion for advancing technology, I am committed to using AI
			for social good and exploring green technology innovations. Let’s collaborate to shape a
			better future through cutting-edge solutions to real-world challenges.
		</p>
	</div>


</div>
<!-- Add the following card -->
<div class="card p-4 mt-8 w-full max-w-3xl mx-auto">
	<div class="flex flex-col md:flex-row items-center gap-4">
		<img src="https://res.cloudinary.com/dltwftrgc/image/upload/v1712504276/Projects/EmbedAnything_500_x_200_px_a4l8xu.png" alt="EmbedAnything Preview" class="w-full md:w-1/3 rounded-lg">
		<div class="flex-1">
			<h3 class="h3 mb-2">EmbedAnything</h3>
			<p class="mb-4">A Rust-based library designed to optimize embedding pipelines, making them faster and more reliable.</p>
			<a href="https://github.com/StarlightSearch/EmbedAnything" target="_blank" rel="noopener noreferrer" class="btn variant-filled-primary">
				<i class="fab fa-github mr-2"></i> View on GitHub
			</a>
		</div>
	</div>
</div>